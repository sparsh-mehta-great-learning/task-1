import streamlit as st
import os
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
import whisper
from openai import OpenAI
import tempfile
from scipy.signal import find_peaks
import gc
import warnings
import re
from contextlib import contextmanager

import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import warnings

class OptimizedAudioAnalyzer:
    def __init__(self):
        self.sr = 4000  # Fixed sample rate
        self.hop_length = 512
        self.n_fft = 1024
        self.chunk_size = 60  # seconds
        
    def _process_chunk(self, audio_path, chunk_idx, duration):
        """Efficiently processes a single chunk of audio."""
        start_time = chunk_idx * self.chunk_size
        dur = min(self.chunk_size, duration - start_time)

        y, _ = librosa.load(audio_path, offset=start_time, duration=dur, sr=self.sr, mono=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, window="hann", center=False))
            rms = np.sqrt(np.mean(stft ** 2, axis=0))

            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=self.sr,
                frame_length=self.n_fft, hop_length=self.hop_length, fill_na=None, center=False
            )

            spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=self.sr, hop_length=self.hop_length)[0]

            return {
                "pitch": f0[voiced_flag],
                "rms": rms,
                "spectral_centroid": spectral_centroid,
            }

    def analyze_audio(self, audio_path):
        """Parallel processing for faster analysis."""
        duration = librosa.get_duration(path=audio_path)
        num_chunks = int(np.ceil(duration / self.chunk_size))

        with ThreadPoolExecutor(max_workers=2) as executor:
            chunk_results = list(executor.map(
                lambda chunk_idx: self._process_chunk(audio_path, chunk_idx, duration), range(num_chunks)
            ))

        pitch_values = np.concatenate([r["pitch"] for r in chunk_results if len(r["pitch"]) > 0])
        rms_values = np.concatenate([r["rms"] for r in chunk_results])
        spectral_centroids = np.concatenate([r["spectral_centroid"] for r in chunk_results])

        pitch_stats = {
            "mean": float(np.nanmean(pitch_values)),
            "std": float(np.nanstd(pitch_values)),
            "range": float(np.nanpercentile(pitch_values, 95) - np.nanpercentile(pitch_values, 5))
        }

        silence_threshold = np.mean(rms_values) * 0.1
        silent_frames = rms_values < silence_threshold
        frame_time = self.hop_length / self.sr

        pause_durations = []
        current_pause = 0
        for is_silent in silent_frames:
            if is_silent:
                current_pause += 1
            elif current_pause > 0:
                duration = current_pause * frame_time
                if duration > 0.3:
                    pause_durations.append(duration)
                current_pause = 0

        pause_stats = {
            "total_pauses": len(pause_durations),
            "mean_pause_duration": float(np.mean(pause_durations)) if pause_durations else 0.0
        }

        return {
            "pitch_analysis": {"statistics": pitch_stats},
            "rhythm_analysis": {"pause_stats": pause_stats},
            "energy_dynamics": {
                "rms_energy_mean": float(np.mean(rms_values)),
                "rms_energy_std": float(np.std(rms_values)),
                "energy_range": float(np.percentile(rms_values, 95) - np.percentile(rms_values, 5))
            },
            "spectral_centroid_mean": float(np.mean(spectral_centroids))
        }

class CPUMentorEvaluator:
    def __init__(self):
        """Initialize the evaluator for CPU usage."""
        self.api_key = st.secrets["OPENAI_API_KEY"]
        if not self.api_key:
            raise ValueError("OpenAI API key not found in secrets")
        
        # Updated OpenAI client initialization - removed proxies argument
        self.client = OpenAI(api_key=self.api_key)
        self.whisper_model = None
        self.accent_classifier = None


    def _clear_memory(self):
        """Clear memory and run garbage collection."""
        if hasattr(self, 'whisper_model') and self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None

        if hasattr(self, 'accent_classifier') and self.accent_classifier is not None:
            del self.accent_classifier
            self.accent_classifier = None

        gc.collect()

    @contextmanager
    def load_whisper_model(self):
        """Load Whisper model with proper memory management."""
        try:
            self._clear_memory()
            self.whisper_model = whisper.load_model("tiny", device="cpu")
            yield self.whisper_model
        finally:
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            gc.collect()

    def extract_audio(self, video_path):
        """Extract audio from video file with optimized settings."""
        temp_audio = None
        video = None
        try:
            self._clear_memory()
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            video = VideoFileClip(video_path, audio=True, target_resolution=(360,None), verbose=False)
            video.audio.write_audiofile(temp_audio.name, fps=8000, verbose=False, logger=None)
            return temp_audio.name
        except Exception as e:
            if temp_audio and os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
            raise Exception(f"Audio extraction failed: {str(e)}")
        finally:
            if video:
                video.close()
            self._clear_memory()

    def analyze_audio_features(self, audio_path):
        analyzer = OptimizedAudioAnalyzer()
        return analyzer.analyze_audio(audio_path)

    def _analyze_pauses(self, silent_frames, frame_time):
        """Analyze pauses with minimal memory usage."""
        pause_durations = []
        current_pause = 0

        for is_silent in silent_frames:
            if is_silent:
                current_pause += 1
            elif current_pause > 0:
                duration = current_pause * frame_time
                if duration > 0.3:  # Only count pauses longer than 300ms
                    pause_durations.append(duration)
                current_pause = 0

        if pause_durations:
            return {
                'total_pauses': len(pause_durations),
                'mean_pause_duration': float(np.mean(pause_durations))
            }
        return {
            'total_pauses': 0,
            'mean_pause_duration': 0.0
        }

    def calculate_speech_metrics(self, transcript, audio_duration):
        """Calculate words per minute and other speech metrics."""
        words = len(transcript.split())
        minutes = audio_duration / 60
        return {
            'words_per_minute': words / minutes if minutes > 0 else 0,
            'total_words': words,
            'duration_minutes': minutes
        }

    def _analyze_voice_quality(self, transcript, audio_features):
        """Analyze voice quality aspects."""
        try:
            prompt = f"""Analyze the following voice metrics for teaching quality:
Transcript excerpt: {transcript[:]}...
Voice Metrics:
- Pitch Mean: {audio_features['pitch_analysis']['statistics']['mean']:.1f}Hz
- Pitch Variation: {audio_features['pitch_analysis']['statistics']['std']:.1f}Hz
- Energy Dynamics: {audio_features['energy_dynamics']['rms_energy_mean']:.2f}
Evaluate voice quality focusing on:
1. Clarity and projection
2. Emotional engagement
3. Professional tone
"""
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in voice analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Voice quality analysis failed: {str(e)}"

    def _analyze_teaching_content(self, transcript):
        """Analyze teaching content for accuracy, principles, and examples."""
        try:
            prompt = f"""Analyze this teaching transcript for:
1. Subject Matter Accuracy:
  - Identify any factual errors, wrong assumptions, or incorrect correlations
  - Rate accuracy on a scale of 0-1
2. First Principles Approach:
  - Evaluate if concepts are built from fundamentals before introducing technical terms
  - Rate approach on a scale of 0-1
3. Examples and Business Context:
  - Assess use of business examples and practical context
  - Rate contextual relevance on a scale of 0-1
Transcript: {transcript}...
Provide specific citations for any identified issues.
"""
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in pedagogical assessment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Teaching content analysis failed: {str(e)}"

    def _analyze_code_explanation(self, transcript):
        """Analyze code explanation quality."""
        try:
            prompt = f"""Analyze the code explanation in this transcript for:
1. Depth of Explanation:
  - Evaluate coverage of syntax, libraries, functions, and methods
  - Rate depth on a scale of 0-1
2. Output Interpretation:
  - Assess business context interpretation of results
  - Rate interpretation on a scale of 0-1
3. Complexity Breakdown:
  - Evaluate explanation of code modules and logical flow
  - Rate breakdown quality on a scale of 0-1
Transcript: {transcript}...
Provide specific citations for any identified issues.
"""
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in code review and teaching."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Code explanation analysis failed: {str(e)}"

    def generate_enhanced_report(self, video_path):
        """Generate structured evaluation report."""
        audio_path = None
        try:
            audio_path = self.extract_audio(video_path)

            with self.load_whisper_model() as model:
                result = model.transcribe(audio_path)
                transcript = result["text"]

            audio_features = self.analyze_audio_features(audio_path)
            audio_duration = librosa.get_duration(path=audio_path)
            speech_metrics = self.calculate_speech_metrics(transcript, audio_duration)

            wpm = speech_metrics['words_per_minute']
            wpm_score = 1 if 120 <= wpm <= 160 else 0

            filler_words = len(re.findall(r'\b(um|uh|like|you know|basically)\b', transcript.lower()))
            fpm = (filler_words / speech_metrics['duration_minutes'])

            ppm = audio_features['rhythm_analysis']['pause_stats']['total_pauses'] / speech_metrics['duration_minutes']
            pause_score = 1 if 2 <= ppm <= 8 else 0

            energy_values = audio_features['energy_dynamics']
            energy_summary = {
                'min': np.percentile([energy_values['rms_energy_mean']], 0),
                'q1': np.percentile([energy_values['rms_energy_mean']], 25),
                'median': np.percentile([energy_values['rms_energy_mean']], 50),
                'q3': np.percentile([energy_values['rms_energy_mean']], 75),
                'max': np.percentile([energy_values['rms_energy_mean']], 100)
            }

            teaching_analysis = self._analyze_teaching_content(transcript)
            code_analysis = self._analyze_code_explanation(transcript)
            voice_quality = self._analyze_voice_quality(transcript, audio_features)

            intonation_score = 1 if (audio_features['pitch_analysis']['patterns']['rising_count'] +
                                   audio_features['pitch_analysis']['patterns']['falling_count']) / speech_metrics['duration_minutes'] > 5 else 0

            energy_score = 1 if (energy_values['rms_energy_std'] / energy_values['rms_energy_mean']) > 0.2 else 0

            report = f"""REPORT
1. COMMUNICATION
    1. Speech Speed:
        - Words per Minute: {wpm:.1f}
        - Score: {wpm_score} (Acceptable range: 120-160 WPM)
    2. Voice Quality:
        {voice_quality}
    3. Fluency:
        - Fillers per Minute: {fpm:.1f}
        - Score: {1 if fpm < 3 else 0}
    4. Break/Flow:
        - Pauses per Minute: {ppm:.1f}
        - Score: {pause_score}
    5. Intonation:
        - Rising patterns: {audio_features['pitch_analysis']['patterns']['rising_count']}
        - Falling patterns: {audio_features['pitch_analysis']['patterns']['falling_count']}
        - Score: {intonation_score}
    6. Energy:
        Five-point summary:
        - Min: {energy_summary['min']:.2f}
        - Q1: {energy_summary['q1']:.2f}
        - Median: {energy_summary['median']:.2f}
        - Q3: {energy_summary['q3']:.2f}
        - Max: {energy_summary['max']:.2f}
        - Score: {energy_score}
2. TEACHING
    1. Content Analysis:
        {teaching_analysis}
    2. Code Explanation:
        {code_analysis}
Full Transcript:
{transcript}
"""
            return report

        except Exception as e:
            raise Exception(f"Report generation failed: {str(e)}")
        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            self._clear_memory()

def create_temp_directory():
    """Create a temporary directory for file processing."""
    temp_dir = tempfile.mkdtemp()
    return temp_dir

def main():
    st.set_page_config(
        page_title="Mentor Demo Review Tool",
        page_icon="🎓",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #1f77b4;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .section-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .score-good { color: #28a745; }
        .score-warning { color: #ffc107; }
        .score-poor { color: #dc3545; }
        .analysis-section {
            margin-top: 20px;
            padding: 15px;
            border-left: 3px solid #1f77b4;
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎓 Mentor Demo Review Tool")

    # Sidebar with instructions
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Upload your teaching video
        2. Wait for analysis to complete
        3. Review the detailed feedback
        4. Download the full report
        
        **Supported Formats:**
        - MP4
        - AVI
        - MOV
        - MKV
        
        **Analysis Includes:**
        - Speech metrics
        - Teaching quality
        - Voice analysis
        - Content evaluation
        """)
        
        st.markdown("---")
        st.markdown("### Privacy Note")
        st.info("Videos are processed securely and deleted immediately after analysis.")

    # Main content
    uploaded_file = st.file_uploader("Upload your teaching video", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_file:
        try:
            if not st.session_state.get('analysis_complete', False):
                with st.status("Analyzing video...", expanded=True) as status:
                    st.write("Saving video file...")
                    temp_dir = create_temp_directory()
                    temp_video_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_video_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    st.write("Initializing analysis...")
                    evaluator = CPUMentorEvaluator()
                    
                    st.write("Generating report...")
                    report = evaluator.generate_enhanced_report(temp_video_path)
                    st.session_state.report_data = report
                    st.session_state.analysis_complete = True
                    
                    status.update(label="Analysis complete!", state="complete", expanded=False)

            if st.session_state.get('analysis_complete', False):
                report = st.session_state.report_data
                
                # Create tabs for organized display
                comm_tab, teach_tab, trans_tab = st.tabs([
                    "📊 Communication Analysis", 
                    "📝 Teaching Evaluation", 
                    "📄 Transcript"
                ])
                
                with comm_tab:
                    st.markdown("## 🎯 Communication Analysis")
                    
                    # Speech Metrics Section
                    st.markdown("### Speech Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    # Extract and display all communication metrics
                    speech_section = re.search(r"1\. COMMUNICATION(.*?)2\. TEACHING", report, re.DOTALL)
                    if speech_section:
                        speech_text = speech_section.group(1)
                        
                        # Speech Speed
                        wpm_match = re.search(r"Words per Minute: (\d+\.?\d*)", speech_text)
                        if wpm_match:
                            wpm = float(wpm_match.group(1))
                            with col1:
                                st.markdown("#### Speech Speed")
                                color = "good" if 120 <= wpm <= 160 else "warning"
                                st.markdown(f'<div class="metric-value score-{color}">{wpm:.1f} WPM</div>', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Target: 120-160 WPM</div>', unsafe_allow_html=True)
                        
                        # Fluency
                        fpm_match = re.search(r"Fillers per Minute: (\d+\.?\d*)", speech_text)
                        if fpm_match:
                            fpm = float(fpm_match.group(1))
                            with col2:
                                st.markdown("#### Fluency")
                                color = "good" if fpm < 3 else "poor"
                                st.markdown(f'<div class="metric-value score-{color}">{fpm:.1f} FPM</div>', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Fillers per Minute</div>', unsafe_allow_html=True)
                        
                        # Pauses
                        ppm_match = re.search(r"Pauses per Minute: (\d+\.?\d*)", speech_text)
                        if ppm_match:
                            ppm = float(ppm_match.group(1))
                            with col3:
                                st.markdown("#### Strategic Pauses")
                                color = "good" if 2 <= ppm <= 8 else "warning"
                                st.markdown(f'<div class="metric-value score-{color}">{ppm:.1f} PPM</div>', unsafe_allow_html=True)
                                st.markdown('<div class="metric-label">Pauses per Minute</div>', unsafe_allow_html=True)

                    # Voice Quality Analysis
                    st.markdown("### 🎤 Voice Quality Analysis")
                    voice_section = re.search(r"Voice Quality:(.*?)3\. Fluency:", report, re.DOTALL)
                    if voice_section:
                        with st.expander("Detailed Voice Analysis", expanded=True):
                            st.markdown(voice_section.group(1).strip())
                    
                    # Intonation Analysis
                    st.markdown("### 📈 Intonation Patterns")
                    intonation_section = re.search(r"5\. Intonation:(.*?)6\. Energy:", report, re.DOTALL)
                    if intonation_section:
                        with st.expander("Intonation Analysis", expanded=True):
                            st.markdown(intonation_section.group(1).strip())
                    
                    # Energy Analysis
                    st.markdown("### ⚡ Energy Profile")
                    energy_section = re.search(r"6\. Energy:(.*?)2\. TEACHING", report, re.DOTALL)
                    if energy_section:
                        with st.expander("Energy Analysis", expanded=True):
                            st.markdown(energy_section.group(1).strip())
                
                with teach_tab:
                    st.markdown("## 📚 Teaching Analysis")
                    
                    # Content Analysis
                    st.markdown("### Content Analysis")
                    content_section = re.search(r"Content Analysis:(.*?)Code Explanation:", report, re.DOTALL)
                    if content_section:
                        with st.expander("Detailed Content Analysis", expanded=True):
                            content_analysis = content_section.group(1).strip()
                            
                            # Parse and display scores
                            accuracy_score = re.search(r"Rate accuracy.*?(\d+\.?\d*)", content_analysis)
                            principles_score = re.search(r"Rate approach.*?(\d+\.?\d*)", content_analysis)
                            context_score = re.search(r"Rate contextual.*?(\d+\.?\d*)", content_analysis)
                            
                            col1, col2, col3 = st.columns(3)
                            if accuracy_score:
                                with col1:
                                    score = float(accuracy_score.group(1))
                                    color = "good" if score >= 0.8 else "warning" if score >= 0.6 else "poor"
                                    st.markdown("#### Content Accuracy")
                                    st.markdown(f'<div class="metric-value score-{color}">{score:.2f}</div>', unsafe_allow_html=True)
                            
                            if principles_score:
                                with col2:
                                    score = float(principles_score.group(1))
                                    color = "good" if score >= 0.8 else "warning" if score >= 0.6 else "poor"
                                    st.markdown("#### First Principles")
                                    st.markdown(f'<div class="metric-value score-{color}">{score:.2f}</div>', unsafe_allow_html=True)
                            
                            if context_score:
                                with col3:
                                    score = float(context_score.group(1))
                                    color = "good" if score >= 0.8 else "warning" if score >= 0.6 else "poor"
                                    st.markdown("#### Business Context")
                                    st.markdown(f'<div class="metric-value score-{color}">{score:.2f}</div>', unsafe_allow_html=True)
                            
                            st.markdown("#### Detailed Analysis")
                            st.markdown(content_analysis)
                    
                    # Code Explanation Analysis
                    st.markdown("### 💻 Code Explanation Quality")
                    code_section = re.search(r"Code Explanation:(.*?)Full Transcript:", report, re.DOTALL)
                    if code_section:
                        with st.expander("Code Teaching Analysis", expanded=True):
                            code_analysis = code_section.group(1).strip()
                            
                            # Parse and display scores
                            depth_score = re.search(r"Rate depth.*?(\d+\.?\d*)", code_analysis)
                            interpretation_score = re.search(r"Rate interpretation.*?(\d+\.?\d*)", code_analysis)
                            breakdown_score = re.search(r"Rate breakdown.*?(\d+\.?\d*)", code_analysis)
                            
                            col1, col2, col3 = st.columns(3)
                            if depth_score:
                                with col1:
                                    score = float(depth_score.group(1))
                                    color = "good" if score >= 0.8 else "warning" if score >= 0.6 else "poor"
                                    st.markdown("#### Explanation Depth")
                                    st.markdown(f'<div class="metric-value score-{color}">{score:.2f}</div>', unsafe_allow_html=True)
                            
                            if interpretation_score:
                                with col2:
                                    score = float(interpretation_score.group(1))
                                    color = "good" if score >= 0.8 else "warning" if score >= 0.6 else "poor"
                                    st.markdown("#### Output Interpretation")
                                    st.markdown(f'<div class="metric-value score-{color}">{score:.2f}</div>', unsafe_allow_html=True)
                            
                            if breakdown_score:
                                with col3:
                                    score = float(breakdown_score.group(1))
                                    color = "good" if score >= 0.8 else "warning" if score >= 0.6 else "poor"
                                    st.markdown("#### Complexity Breakdown")
                                    st.markdown(f'<div class="metric-value score-{color}">{score:.2f}</div>', unsafe_allow_html=True)
                            
                            st.markdown("#### Detailed Analysis")
                            st.markdown(code_analysis)
                
                with trans_tab:
                    st.markdown("## 📝 Full Transcript")
                    transcript_section = re.search(r"Full Transcript:(.*?)(?=\Z)", report, re.DOTALL)
                    if transcript_section:
                        st.markdown(transcript_section.group(1).strip())
                
                # Download button
                st.download_button(
                    label="📥 Download Full Report",
                    data=report,
                    file_name="mentor_analysis_report.txt",
                    mime="text/plain",
                    help="Download the complete analysis report including all metrics and recommendations"
                )

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.error("Please try uploading the video again or contact support if the issue persists.")

        finally:
            # Cleanup
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
            gc.collect()

if __name__ == "__main__":
    main()
