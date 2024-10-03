import streamlit as st
import os
import openai
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.chains import LLMChain
from PIL import Image
import requests
import io
import numpy as np
import math
from moviepy.editor import *
from moviepy.audio.io.AudioFileClip import AudioFileClip
import tempfile
import time
import asyncio
import aiohttp
from datetime import datetime

# Always require OpenAI API key at start
st.title("Video Automation App")

api_key = st.text_input("Please enter your OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key
    # Save API key to .env file
    with open(".env", "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    st.success("API Key saved to .env file")
else:
    st.error("OpenAI API Key is required to proceed.")
    st.stop()

def transcribe_audio(audio_paths, response_format="srt"):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai.api_key}"
    }
    data = {
        "model": "whisper-1",
        "response_format": response_format
    }
    try:
        all_transcripts = []
        for audio_path in audio_paths:
            with open(audio_path, 'rb') as audio_file:
                files = {
                    "file": (os.path.basename(audio_path), audio_file, "application/octet-stream")
                }
                response = requests.post(url, headers=headers, data=data, files=files)
                response.raise_for_status()
                if response_format in ["json", "verbose_json"]:
                    all_transcripts.append(response.json())
                else:
                    all_transcripts.append(response.text)
        return all_transcripts
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def parse_transcript(transcript_responses, response_format):
    parsed_transcripts = []
    total_duration = 0
    
    for transcript_response in transcript_responses:
        if response_format == "srt":
            # Parse SRT formatted transcript
            srt_string = transcript_response
            srt_parts = srt_string.strip().split('\n\n')
            for part in srt_parts:
                lines = part.strip().split('\n')
                if len(lines) >= 3:
                    time_range = lines[1]
                    text = ' '.join(lines[2:])
                    start_time_str, end_time_str = time_range.split(' --> ')
                    # Convert time strings to seconds and adjust for total duration
                    start_time = time_str_to_seconds(start_time_str) + total_duration
                    end_time = time_str_to_seconds(end_time_str) + total_duration
                    parsed_transcripts.append((start_time, end_time, text))
            # Update total duration
            total_duration = parsed_transcripts[-1][1]
        elif response_format in ["json", "verbose_json"]:
            # Parse JSON formatted transcript
            for segment in transcript_response['segments']:
                start_time = segment['start'] + total_duration
                end_time = segment['end'] + total_duration
                text = segment['text'].strip()
                parsed_transcripts.append((start_time, end_time, text))
            # Update total duration
            total_duration += transcript_response['segments'][-1]['end']
        elif response_format == "text":
            # Simple text transcript without timestamps
            text = transcript_response.strip()
            parsed_transcripts.append((total_duration, None, text))
            # Update total duration (assuming 1 second per character as a rough estimate)
            total_duration += len(text) / 10
    
    return parsed_transcripts

def time_str_to_seconds(time_str):
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    return total_seconds

def generate_timestamped_prompts(transcript, user_prompt, interval, total_duration):
    llm = LangChainOpenAI(temperature=0.7, openai_api_key=openai.api_key)
    
    # Calculate the number of prompts
    num_prompts = math.ceil(total_duration / interval)
    
    prompt_template = PromptTemplate(
        input_variables=["text", "user_prompt", "interval", "total_duration", "num_prompts"],
        template="""
Given the following transcript, a {interval}-second interval, and a total duration of {total_duration} seconds, generate image prompts:

Transcript: {text}

User prompt: {user_prompt}

Create exactly {num_prompts} AI image prompts to generate highly detailed, cinematic CGI images in 4K. Do not add any words or branding to the images. Avoid image prompts that focus on any specific person.

Ensure that all prompts comply with DALL·E content policy to avoid any disallowed content.

Format your response as follows:
[Start time-End time] Image prompt

Ensure that the timestamps cover the entire duration of the audio, adjusting the last timestamp if necessary.

Begin generating prompts now:
"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Combine all text from the transcript
    full_text = " ".join([text for _, _, text in transcript])
    
    result = chain.run(
        text=full_text, 
        user_prompt=user_prompt, 
        interval=interval, 
        total_duration=total_duration,
        num_prompts=num_prompts
    )
    
    # Parse the result into a list of tuples (timestamp, prompt)
    parsed_result = []
    lines = result.strip().split('\n')
    for line in lines:
        if line.startswith('[') and ']' in line:
            timestamp, prompt = line.split(']', 1)
            timestamp = timestamp.strip('[]')
            prompt = prompt.strip()
            parsed_result.append((timestamp, prompt))
    
    return parsed_result

def get_timestamp_filename(index):
    """Generate a filename with the current date and time in the specified format."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"generated_image_{timestamp}_{index}.png"

async def generate_image_batch(prompts, target_width, target_height):
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    # Create 'images' folder if it doesn't exist
    os.makedirs('images', exist_ok=True)

    async def generate_single_image(session, prompt, index):
        data = {
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024"  # Request maximum size
        }

        try:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                image_url = result["data"][0]["url"]
                async with session.get(image_url) as img_response:
                    img_data = await img_response.read()
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    img = img.resize((target_width, target_height), Image.LANCZOS)
                    
                    # Save the image to the 'images' folder with the new timestamp format
                    try:
                        img_filename = get_timestamp_filename(index)
                        img_path = os.path.join('images', img_filename)
                        img.save(img_path)
                        st.success(f"Image saved: {img_path}")
                    except Exception as save_error:
                        st.warning(f"Failed to save image {index}: {str(save_error)}")
                    
                    return img
        except Exception as e:
            st.error(f"Error generating image {index}: {str(e)}")
            return None

    async with aiohttp.ClientSession() as session:
        tasks = [generate_single_image(session, prompt, i) for i, prompt in enumerate(prompts)]
        return await asyncio.gather(*tasks)
    
def create_video(images, audio_paths, durations, target_width, target_height, fps=30):
    clips = []
    cumulative_duration = 0
    
    for image, duration in zip(images, durations):
        img_array = np.array(image)
        clip = ImageClip(img_array).set_duration(duration)
        clip = clip.set_start(cumulative_duration)
        clips.append(clip)
        cumulative_duration += duration
    
    video = CompositeVideoClip(clips, size=(target_width, target_height))
    
    # Concatenate audio files
    audio_clips = [AudioFileClip(audio_path) for audio_path in audio_paths]
    final_audio = concatenate_audioclips(audio_clips)
    
    final_clip = video.set_audio(final_audio)
    final_clip = final_clip.set_fps(fps)
    
    return final_clip

async def generate_image_batch_leonardo(prompts, target_width, target_height, api_key):
    url = "https://cloud.leonardo.ai/api/rest/v1/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Create 'images' folder if it doesn't exist
    os.makedirs('images', exist_ok=True)

    async def generate_single_image(session, prompt, index):
        data = {
            "prompt": prompt,
            "num_images": 1,
            "width": 1024,
            "height": 1024,
            "modelId": "ac614f96-1082-45bf-be9d-757f2d31c174"
        }

        try:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                generation_id = result["sdGenerationJob"]["generationId"]

                status_url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}"
                while True:
                    async with session.get(status_url, headers=headers) as status_response:
                        status_result = await status_response.json()
                        if status_result["generations_by_pk"]["status"] == "COMPLETE":
                            image_url = status_result["generations_by_pk"]["generated_images"][0]["url"]
                            break
                        await asyncio.sleep(1)

                async with session.get(image_url) as img_response:
                    img_data = await img_response.read()
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                    img = img.resize((target_width, target_height), Image.LANCZOS)
                    
                    # Save the image to the 'images' folder with the new timestamp format
                    try:
                        img_filename = get_timestamp_filename(index)
                        img_path = os.path.join('images', img_filename)
                        img.save(img_path)
                        st.success(f"Image saved: {img_path}")
                    except Exception as save_error:
                        st.warning(f"Failed to save image {index}: {str(save_error)}")
                    
                    return img
        except Exception as e:
            st.error(f"Error generating image {index}: {str(e)}")
            return None

    async with aiohttp.ClientSession() as session:
        tasks = [generate_single_image(session, prompt, i) for i, prompt in enumerate(prompts)]
        return await asyncio.gather(*tasks)

def main():
    # Initialize session state variables
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []
    if 'durations' not in st.session_state:
        st.session_state.durations = []
    if 'image_prompts' not in st.session_state:
        st.session_state.image_prompts = []
    if 'video_created' not in st.session_state:
        st.session_state.video_created = False
    if 'output_path' not in st.session_state:
        st.session_state.output_path = None
    
    # Select response format for transcription
    response_format = st.selectbox("Select transcription response format", ["srt", "json", "text", "verbose_json", "vtt"])
    image_service = st.selectbox("Select Image Generation Service", ["OpenAI", "Leonardo AI"])

    # If Leonardo AI is selected, require Leonardo API key
    if image_service == "Leonardo AI":
        leonardo_api_key = st.text_input("Please enter your Leonardo AI API Key", type="password")
        if not leonardo_api_key:
            st.error("Leonardo API Key is required for image generation.")
            st.stop()
            
    uploaded_files = st.file_uploader("Upload audio files", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
    user_prompt = st.text_input("Optional: Enter a prompt to guide image generation", "")
    interval = st.number_input("Enter interval for image change (in seconds)", min_value=1, value=10)
    # Image resolution options
    image_resolution = st.selectbox("Select image resolution", ["1024x576 (16:9)", "576x1024 (9:16)"])
    
    if image_resolution == "1024x576 (16:9)":
        target_width, target_height = 1024, 576
        aspect_ratio = "16:9"
    else:
        target_width, target_height = 576, 1024
        aspect_ratio = "9:16"
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.audio(uploaded_file)
    
        if st.button("Start Automation"):
            temp_audio_paths = []
            audio_clips = []
            try:
                # Create temporary files to store the uploaded audios
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_audio_paths.append(tmp_file.name)
    
                with st.spinner("Transcribing audio..."):
                    transcript_responses = transcribe_audio(temp_audio_paths, response_format=response_format)
                    if transcript_responses:
                        transcript = parse_transcript(transcript_responses, response_format)
                        st.success("Audio transcribed successfully!")
                        st.write(f"Number of transcript segments: {len(transcript)}")
                    else:
                        st.error("Failed to transcribe audio.")
                        return
    
                total_duration = sum([AudioFileClip(path).duration for path in temp_audio_paths])
                st.write(f"Total audio duration: {total_duration:.2f} seconds")
    
                with st.spinner("Generating image prompts..."):
                    image_prompts = generate_timestamped_prompts(transcript, user_prompt, interval, total_duration)
                    st.success("Image prompts generated successfully!")
                    st.write(f"Number of image prompts: {len(image_prompts)}")
                    for timestamp, prompt in image_prompts:
                        st.write(f"[{timestamp}] {prompt}")
                    # Store prompts in session state
                    st.session_state.image_prompts = image_prompts
    
                with st.spinner("Generating images..."):
                    prompts = [prompt for _, prompt in image_prompts]
                    if image_service == 'OpenAI':
                        generated_images = asyncio.run(generate_image_batch(prompts, target_width, target_height))
                    else:
                        generated_images = asyncio.run(generate_image_batch_leonardo(prompts, target_width, target_height, leonardo_api_key))
                    
                    durations = []
                    for i, image in enumerate(generated_images):
                        if image:
                            duration = min(interval, total_duration - i * interval)
                            durations.append(duration)
                            st.image(image, caption=f"Image for {image_prompts[i][0]}")
                        else:
                            st.warning(f"Failed to generate image for timestamp {image_prompts[i][0]}")
                    
                    st.success(f"Images generated successfully! Total: {len(generated_images)}")
                    # Store images and durations in session state
                    st.session_state.generated_images = generated_images
                    st.session_state.durations = durations
    
                if not generated_images:
                    st.error("No images were generated. Cannot create video.")
                    return
    
                with st.spinner("Creating video..."):
                    try:
                        final_clip = create_video(generated_images, temp_audio_paths, durations, target_width, target_height, fps=30)
                        # Save video to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                            output_path = tmp_video.name
                        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30)
                        st.success("Video created successfully!")
                        # Store output path in session state
                        st.session_state.output_path = output_path
                        st.session_state.video_created = True
    
                        # Display the video
                        st.video(output_path)
    
                        # Provide a download button
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Video",
                                data=file.read(),
                                file_name="output_video.mp4",
                                mime="video/mp4"
                            )
                    except Exception as e:
                        st.error(f"Error during video creation: {str(e)}")
                        st.write("Debug information:")
                        st.write(f"Number of generated images: {len(generated_images)}")
                        st.write(f"Durations: {durations}")
            finally:
                # Clean up resources
                for temp_audio_path in temp_audio_paths:
                    if os.path.exists(temp_audio_path):
                        for _ in range(5):  # Try up to 5 times
                            try:
                                os.unlink(temp_audio_path)
                                break
                            except PermissionError:
                                time.sleep(1)  # Wait for 1 second before trying again
                        else:
                            st.warning("Could not delete temporary audio file. It will be deleted when the system restarts.")
        else:
            # If video was already created, display it
            if st.session_state.video_created and st.session_state.output_path:
                st.video(st.session_state.output_path)
                # Provide a download button
                with open(st.session_state.output_path, "rb") as file:
                    st.download_button(
                        label="Download Video",
                        data=file.read(),
                        file_name="output_video.mp4",
                        mime="video/mp4"
                    )
            # Display generated images and prompts if they exist
            if st.session_state.generated_images:
                st.write("Generated Images:")
                for image, prompt in zip(st.session_state.generated_images, st.session_state.image_prompts):
                    st.image(image, caption=prompt[1])
            if st.session_state.image_prompts:
                st.write("Image Prompts:")
                for timestamp, prompt in st.session_state.image_prompts:
                    st.write(f"[{timestamp}] {prompt}")

if __name__ == "__main__":
    main()
