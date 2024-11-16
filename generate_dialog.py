import os
import random
import json
from pydub import AudioSegment
from tools.file import audio_to_bytes, read_ref_text
from tools.schema import ServeReferenceAudio, ServeTTSRequest
import requests
import ormsgpack
from tqdm import tqdm  # 导入 tqdm


# 从 reference 目录读取子文件夹
def get_reference_ids(reference_dir):
    return [folder for folder in os.listdir(reference_dir) if os.path.isdir(os.path.join(reference_dir, folder))]


# 根据对话生成音频
def generate_audio_for_conversation(conversation, reference_dir, output_dir):
    reference_ids = get_reference_ids(reference_dir)
    
    # 获取对话 ID 和保存路径
    conversation_id = conversation['id']
    conversation_dir = os.path.join(output_dir, conversation_id)
    
    # 如果该对话的文件夹已经存在且包含 .json 文件，跳过
    if os.path.exists(conversation_dir) and os.path.isfile(os.path.join(conversation_dir, f"{conversation_id}.json")):
        print(f"Skipping already processed conversation {conversation_id}")
        return
    
    # 随机为 user 和 assistant 选择一个 reference_id
    user_reference_id = random.choice(reference_ids)
    assistant_reference_id = random.choice(reference_ids)
    
    # 保存每一轮对话的音频文件
    os.makedirs(conversation_dir, exist_ok=True)
    
    user_turns = []
    assistant_turns = []
    
    # 将用户和助手的每个轮次分别收集
    for idx, exchange in enumerate(conversation['data']):
        if 'user' in exchange:
            user_turns.append(exchange['user'])
        if 'assistant' in exchange:
            assistant_turns.append(exchange['assistant'])
    
    # 生成 user 和 assistant 的音频
    user_audio_files = generate_audio(user_turns, user_reference_id, conversation_id, conversation_dir, "user")
    assistant_audio_files = generate_audio(assistant_turns, assistant_reference_id, conversation_id, conversation_dir, "assistant")
    
    # 如果生成音频失败，则跳过后续处理
    if not user_audio_files or not assistant_audio_files:
        print(f"Skipping conversation {conversation_id} due to failed audio generation.")
        return
    
    # 拼接音频并保存
    merge_audio_files(conversation_id, conversation_dir, user_audio_files, assistant_audio_files)
    
    # 保存 JSON 文件
    save_json(conversation_id, conversation_dir, user_turns, assistant_turns, user_reference_id, assistant_reference_id)


# 生成音频
def generate_audio(turns, reference_id, conversation_id, conversation_dir, role):
    audio_files = []
    
    # 使用 tqdm 显示进度条
    for idx, text in tqdm(enumerate(turns), total=len(turns), desc=f"Generating {role} audio"):
        use_memory_cache = "never" if idx == 0 else "on-demand"
        audio_file_name = f"{role}_turn_{idx + 1}.wav"
        audio_file_path = os.path.join(conversation_dir, audio_file_name)
        
        # 请求生成语音
        success = generate_tts(text, reference_id, use_memory_cache, audio_file_path)
        if success:
            audio_files.append(audio_file_path)
        else:
            # 如果生成音频失败，则停止并跳过后续处理
            return []
    
    return audio_files


# 生成 TTS 请求
def generate_tts(text, reference_id, use_memory_cache, output_path):
    args = {
        'text': text,
        'reference_id': reference_id,
        'use_memory_cache': use_memory_cache,
        'output': output_path,
        'format': 'wav',
        'play': False
    }
    
    data = {
        'text': args['text'],
        'references': [],
        'reference_id': args['reference_id'],
        'normalize': True,
        'format': args['format'],
        'use_memory_cache': args['use_memory_cache']
    }
    
    pydantic_data = ServeTTSRequest(**data)
    
    response = requests.post(
        "http://127.0.0.1:8080/v1/tts",  # 本地服务器 URL
        data=ormsgpack.packb(pydantic_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers={"authorization": "Bearer YOUR_API_KEY", "content-type": "application/msgpack"}
    )
    
    if response.status_code == 200:
        audio_content = response.content
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_content)
        return True
    else:
        # 如果请求失败（如 500 错误），保存失败信息并返回 False
        error_message = f"Error generating audio for text '{text}': {response.status_code} - {response.text}\nData:\n{data}"
        error_file_path = os.path.join(os.path.dirname(output_path), "error_log.txt")
        with open(error_file_path, "a") as error_file:
            error_file.write(error_message + "\n")
        print(f"Error during audio generation for conversation. Details saved to {error_file_path}")
        return False


# 拼接音频文件（交替拼接 user 和 assistant 的音频）
def merge_audio_files(conversation_id, conversation_dir, user_audio_files, assistant_audio_files):
    combined = AudioSegment.empty()
    
    # 使用 tqdm 显示拼接进度
    for user_audio, assistant_audio in tqdm(zip(user_audio_files, assistant_audio_files), 
                                             total=len(user_audio_files), 
                                             desc="Merging audio files"):
        user_audio_segment = AudioSegment.from_wav(user_audio)
        assistant_audio_segment = AudioSegment.from_wav(assistant_audio)
        
        combined += user_audio_segment  # 先加用户的
        # 可选：添加间隔（如 500 毫秒静音）
        combined += AudioSegment.silent(duration=500)  
        combined += assistant_audio_segment  # 再加助手的
        # 可选：再添加间隔
        combined += AudioSegment.silent(duration=500)
    
    # 如果剩下还有 user 或 assistant 的音频，则继续拼接
    remaining_audio_files = user_audio_files[len(assistant_audio_files):] or assistant_audio_files[len(user_audio_files):]
    for audio_file in tqdm(remaining_audio_files, desc="Merging remaining audio files"):
        audio_segment = AudioSegment.from_wav(audio_file)
        combined += audio_segment
    
    combined_output_path = os.path.join(conversation_dir, f"{conversation_id}_combined.wav")
    combined.export(combined_output_path, format="wav")
    print(f"Combined audio saved at {combined_output_path}")


# 保存 JSON 文件
def save_json(conversation_id, conversation_dir, user_turns, assistant_turns, user_reference_id, assistant_reference_id):
    json_data = {
        "conversation_id": conversation_id,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "user_reference_id": user_reference_id,
        "assistant_reference_id": assistant_reference_id
    }
    
    json_file_path = os.path.join(conversation_dir, f"{conversation_id}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"JSON saved at {json_file_path}")


# 主函数
def main(jsonl_file_path, reference_dir, output_dir):
    with open(jsonl_file_path, 'r') as file:
        for line in tqdm(file, desc="Processing conversations"):
            conversation = json.loads(line.strip())
            generate_audio_for_conversation(conversation, reference_dir, output_dir)

# 调用主函数
if __name__ == "__main__":
    jsonl_file_path = "/gyfs_cq_qqgc/donghuayu/TTS/speech_dialogue_5k.jsonl"  # 指定 JSONL 文件路径
    reference_dir = "/gyfs_cq_qqgc/donghuayu/TTS/fish-speech/references"  # 指定 reference 目录路径
    output_dir = "/gyfs_cq_qqgc/donghuayu/TTS/dialogs"  # 指定输出目录
    main(jsonl_file_path, reference_dir, output_dir)
