import os
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline
from openai import OpenAI
import tempfile
import streamlit as st
import pandas as pd

def extract_audio(video_path):
    """Extract audio from video file."""
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio.name)
    return temp_audio.name

def transcribe_audio(audio_path):
    """Transcribe audio to text using Whisper."""
    model = whisper.load_model("base")  # Use "small", "medium", or "large" for higher accuracy
    result = model.transcribe(audio_path)
    return result.get("text", "Whisper transcription failed")

def analyze_accent(client, text):
    """Analyze accent and mother tongue influence."""
    prompt = f"""
    Analyze the following text for mother tongue influence and suggest 
    suitable regions for teaching based on the language patterns:

    Text: {text}

    Consider:
    1. Language patterns
    2. Word usage
    3. Sentence structure
    4. Common linguistic markers

    Provide a detailed analysis and regional recommendations.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in linguistic analysis"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def evaluate_subject_expertise(client, text):
    """Evaluate subject matter expertise."""
    prompt = f"""
    Evaluate the following teaching session transcript for subject matter expertise:

    Text: {text}

    Analyze:
    1. Technical accuracy
    2. Depth of knowledge
    3. Clarity of explanations
    4. Use of examples
    5. Handling of complex concepts

    Provide a detailed evaluation with specific examples and a score out of 10.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in educational assessment"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def evaluate_communication(client, text):
    """Evaluate communication skills."""
    prompt = f"""
    Evaluate the following teaching session transcript for communication skills:

    Text: {text}

    Analyze:
    1. Clarity of expression
    2. Engagement level
    3. Pace and flow
    4. Use of voice modulation (based on transcript patterns)
    5. Interactive elements

    Provide a detailed evaluation with specific examples and a score out of 10.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in communication assessment"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_report(client, video_path):
    """Generate comprehensive evaluation report."""
    # Extract and transcribe audio
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)

    # Perform evaluations
    accent_analysis = analyze_accent(client, transcript)
    subject_expertise = evaluate_subject_expertise(client, transcript)
    communication_eval = evaluate_communication(client, transcript)

    # Generate report
    report = f"""
    MENTOR EVALUATION REPORT
    =======================

    1. DELIVERY ANALYSIS
    -------------------
    A. Mother Tongue Influence and Regional Suitability
    {accent_analysis}

    B. Communication Assessment
    {communication_eval}

    2. SUBJECT MATTER EXPERTISE
    -------------------------
    {subject_expertise}

    Raw Transcript:
    --------------
    {transcript}
    """

    # Prepare tabular data
    evaluation_results = pd.DataFrame([
        {"Category": "Accent Analysis", "Details": accent_analysis},
        {"Category": "Subject Expertise", "Details": subject_expertise},
        {"Category": "Communication", "Details": communication_eval}
    ])

    # Clean up temporary files
    os.unlink(audio_path)

    return report, evaluation_results, transcript

# Streamlit UI
def main():
    st.title("Mentor Evaluation System")

    # Input OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        return

    client = OpenAI(api_key=api_key)

    # Upload video file
    video_file = st.file_uploader("Upload a Mentor Demo Video", type=["mp4", "mov", "avi"])

    if video_file:
        with st.spinner("Processing video and generating report..."):
            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            with open(temp_video_path, 'wb') as f:
                f.write(video_file.read())

            # Generate report
            try:
                report, evaluation_results, transcript = generate_report(client, temp_video_path)
                st.success("Report generated successfully!")

                # Display report
                st.text_area("Evaluation Report", report, height=300)

                # Display tabular results
                st.subheader("Evaluation Results")
                st.dataframe(evaluation_results)

                # Option to download report
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="mentor_evaluation_report.txt",
                    mime="text/plain"
                )

                # Option to download transcript
                st.download_button(
                    label="Download Transcript",
                    data=transcript,
                    file_name="mentor_session_transcript.txt",
                    mime="text/plain"
                )

                # Display requirements.txt
                st.subheader("Dependencies")
                requirements = """\
SpeechRecognition
moviepy
openai
transformers
torch
pydub
pyaudio
git+https://github.com/openai/whisper.git
                """
                st.text_area("Requirements.txt", requirements, height=150)

                st.download_button(
                    label="Download Requirements.txt",
                    data=requirements,
                    file_name="requirements.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                os.unlink(temp_video_path)

if __name__ == "__main__":
    main()
