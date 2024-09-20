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

def transcribe_audio(audio_path, response_format="srt"):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai.api_key}"
    }
    data = {
        "model": "whisper-1",
        "response_format": response_format
    }
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {
                "file": (os.path.basename(audio_path), audio_file, "application/octet-stream")
            }
            response = requests.post(url, headers=headers, data=data, files=files)
            response.raise_for_status()
            if response_format in ["json", "verbose_json"]:
                return response.json()
            else:
                return response.text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def parse_transcript(transcript_response, response_format):
    if response_format == "srt":
        # Parse SRT formatted transcript
        srt_string = transcript_response
        srt_parts = srt_string.strip().split('\n\n')
        parsed_transcript = []
        for part in srt_parts:
            lines = part.strip().split('\n')
            if len(lines) >= 3:
                time_range = lines[1]
                text = ' '.join(lines[2:])
                start_time_str, end_time_str = time_range.split(' --> ')
                # Convert time strings to seconds
                start_time = time_str_to_seconds(start_time_str)
                end_time = time_str_to_seconds(end_time_str)
                parsed_transcript.append((start_time, end_time, text))
        return parsed_transcript
    elif response_format in ["json", "verbose_json"]:
        # Parse JSON formatted transcript
        parsed_transcript = []
        for segment in transcript_response['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            parsed_transcript.append((start_time, end_time, text))
        return parsed_transcript
    elif response_format == "text":
        # Simple text transcript without timestamps
        text = transcript_response.strip()
        return [(0, None, text)]
    else:
        st.error(f"Unsupported response format: {response_format}")
        return []

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

def generate_image(prompt, target_width, target_height):
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"  # Request maximum size
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        image_url = response.json()["data"][0]["url"]
        image_response = requests.get(image_url)
        img = Image.open(io.BytesIO(image_response.content)).convert('RGB')
        
        # Resize or crop the image to the desired resolution
        img = img.resize((target_width, target_height), Image.LANCZOS)
        return img
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def create_video(images, audio_path, durations, target_width, target_height, fps=30):
    clips = []
    cumulative_duration = 0
    
    for image, duration in zip(images, durations):
        img_array = np.array(image)
        clip = ImageClip(img_array).set_duration(duration)
        clip = clip.set_start(cumulative_duration)
        clips.append(clip)
        cumulative_duration += duration
    
    video = CompositeVideoClip(clips, size=(target_width, target_height))
    audio = AudioFileClip(audio_path)
    final_clip = video.set_audio(audio)
    final_clip = final_clip.set_fps(fps)
    
    return final_clip

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
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
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
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
    
        if st.button("Start Automation"):
            temp_audio_path = None
            audio_clip = None
            try:
                # Create a temporary file to store the uploaded audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_audio_path = tmp_file.name
    
                with st.spinner("Transcribing audio..."):
                    transcript_response = transcribe_audio(temp_audio_path, response_format=response_format)
                    if transcript_response:
                        transcript = parse_transcript(transcript_response, response_format)
                        st.success("Audio transcribed successfully!")
                        st.write(f"Number of transcript segments: {len(transcript)}")
                    else:
                        st.error("Failed to transcribe audio.")
                        return
    
                audio = AudioFileClip(temp_audio_path)
                total_duration = audio.duration
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
                    generated_images = []
                    durations = []
                    for i, (timestamp, prompt) in enumerate(image_prompts):
                        if i * interval >= total_duration:
                            break
                        image = generate_image(prompt, target_width, target_height)
                        if image:
                            generated_images.append(image)
                            duration = min(interval, total_duration - i * interval)
                            durations.append(duration)
                            st.image(image, caption=f"Image for {timestamp}")
                        else:
                            st.warning(f"Failed to generate image for timestamp {timestamp}")
                    st.success(f"Images generated successfully! Total: {len(generated_images)}")
                    # Store images and durations in session state
                    st.session_state.generated_images = generated_images
                    st.session_state.durations = durations
    
                if not generated_images:
                    st.error("No images were generated. Cannot create video.")
                    return
    
                with st.spinner("Creating video..."):
                    try:
                        audio_clip = AudioFileClip(temp_audio_path)
                        final_clip = create_video(generated_images, temp_audio_path, durations, target_width, target_height, fps=30)
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
                if audio_clip:
                    audio_clip.close()
                if temp_audio_path and os.path.exists(temp_audio_path):
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
