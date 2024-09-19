import streamlit as st
import os
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.chains import LLMChain
from PIL import Image
import requests
import toml
import io
import numpy as np
import math
from moviepy.editor import *
from moviepy.audio.io.AudioFileClip import AudioFileClip
import tempfile
import time

# Load API key
def load_api_key(secret_file_path='secrets.toml'):
    try:
        secrets = toml.load(secret_file_path)
        return secrets['OPENAI_API_KEY']
    except Exception as e:
        st.error(f"Error loading API key: {e}")
        return None

OPENAI_API_KEY = load_api_key()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(audio_path):
    try:
        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="srt"
            )
        return transcript
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def parse_srt(srt_string):
    srt_parts = srt_string.strip().split('\n\n')
    parsed_srt = []
    for part in srt_parts:
        lines = part.split('\n')
        if len(lines) >= 3:
            time_range = lines[1]
            text = ' '.join(lines[2:])
            start_time, end_time = time_range.split(' --> ')
            parsed_srt.append((start_time, end_time, text))
    return parsed_srt

def generate_timestamped_prompts(transcript, user_prompt, interval, total_duration):
    llm = LangChainOpenAI(temperature=0.7, api_key=OPENAI_API_KEY)
    
    # Calculate the number of prompts
    num_prompts = math.ceil(total_duration / interval)
    
    prompt_template = PromptTemplate(
        input_variables=["text", "user_prompt", "interval", "total_duration", "num_prompts"],
        template="""
        Given the following transcript, a {interval}-second interval, and a total duration of {total_duration} seconds, generate image prompts:

        Transcript: {text}

        User prompt: {user_prompt}

        Create exactly {num_prompts} AI image prompts to generate highly detailed, cinematic CGI images in 4K. Do not add any words or branding to the images. Avoid image prompts that focus on any specific person.

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
    for i in range(num_prompts):
        start_time = i * interval
        end_time = min((i + 1) * interval, total_duration)
        timestamp = f"{start_time:.2f}-{end_time:.2f}"
        
        # Find the corresponding prompt in the result
        prompt_start = result.find(f"[{start_time}")
        prompt_end = result.find("\n", prompt_start)
        if prompt_start != -1 and prompt_end != -1:
            prompt = result[prompt_start:prompt_end].split("]", 1)[1].strip()
        else:
            prompt = f"Generated image for timestamp {timestamp}"
        
        parsed_result.append((timestamp, prompt))
    
    return parsed_result

def generate_image(prompt, size="1024x1024"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="hd",
            n=1,
        )
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        return Image.open(io.BytesIO(image_response.content))
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def create_video(images, audio_path, durations, aspect_ratio, fps=30):
    clips = []
    cumulative_duration = 0
    
    if aspect_ratio == "16:9":
        target_width, target_height = 3840, 2160  # 4K resolution
    else:  # 9:16
        target_width, target_height = 2160, 3840  # 4K vertical video
    
    for image, duration in zip(images, durations):
        img_array = np.array(image)
        
        # Resize and pad the image
        h, w = img_array.shape[:2]
        aspect = w / h
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            new_w = target_width
            new_h = int(new_w / aspect)
            img_resized = Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS)
            img_padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            img_padded.paste(img_resized, ((target_width - new_w) // 2, (target_height - new_h) // 2))
        else:
            new_h = target_height
            new_w = int(new_h * aspect)
            img_resized = Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS)
            img_padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            img_padded.paste(img_resized, ((target_width - new_w) // 2, (target_height - new_h) // 2))
        
        clip = ImageClip(np.array(img_padded)).set_duration(duration)
        clip = clip.set_start(cumulative_duration)
        clips.append(clip)
        cumulative_duration += duration

    video = CompositeVideoClip(clips, size=(target_width, target_height))
    audio = AudioFileClip(audio_path)
    final_clip = video.set_audio(audio)
    final_clip = final_clip.set_fps(fps)
    
    return final_clip

def main():
    st.title("Video Automation App")

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    user_prompt = st.text_input("Optional: Enter a prompt to guide image generation", "")
    interval = st.number_input("Enter interval for image change (in seconds)", min_value=1, value=10)
    image_size = st.selectbox("Select image size", ["1024x1024", "1792x1024", "1024x1792"])
    aspect_ratio = st.selectbox("Select video aspect ratio", ["16:9", "9:16"])

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
                    transcript_srt = transcribe_audio(temp_audio_path)
                    if transcript_srt:
                        transcript = parse_srt(transcript_srt)
                        st.success("Audio transcribed successfully!")
                        st.write(f"Number of transcript segments: {len(transcript)}")
                    else:
                        st.error("Failed to transcribe audio.")
                        return

                audio = AudioFileClip(temp_audio_path)
                total_duration = audio.duration
                st.write(f"Total audio duration: {total_duration} seconds")

                with st.spinner("Generating image prompts..."):
                    image_prompts = generate_timestamped_prompts(transcript, user_prompt, interval, total_duration)
                    st.success("Image prompts generated successfully!")
                    st.write(f"Number of image prompts: {len(image_prompts)}")
                    for timestamp, prompt in image_prompts:
                        st.write(f"[{timestamp}] {prompt}")

                with st.spinner("Generating images..."):
                    generated_images = []
                    durations = []
                    for i, (timestamp, prompt) in enumerate(image_prompts):
                        if i * interval >= total_duration:
                            break
                        image = generate_image(prompt, size=image_size)
                        if image:
                            generated_images.append(image)
                            duration = min(interval, total_duration - i * interval)
                            durations.append(duration)
                            st.image(image, caption=f"Image for {timestamp}")
                        else:
                            st.warning(f"Failed to generate image for timestamp {timestamp}")
                    st.success(f"Images generated successfully! Total: {len(generated_images)}")

                if not generated_images:
                    st.error("No images were generated. Cannot create video.")
                    return

                with st.spinner("Creating video..."):
                    try:
                        audio_clip = AudioFileClip(temp_audio_path)
                        final_clip = create_video(generated_images, temp_audio_path, durations, aspect_ratio, fps=30)
                        output_path = "output_video.mp4"
                        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=30, bitrate="20000k")
                        st.success("Video created successfully!")

                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Video",
                                data=file,
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

if __name__ == "__main__":
    main()

