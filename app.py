import streamlit as st
import torch
import os
import numpy as np
import librosa
import whisper
from openai import OpenAI
import tempfile
import warnings
import re
from contextlib import contextmanager
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import subprocess
import json
import shutil
from pathlib import Path
import time
from faster_whisper import WhisperModel
import soundfile as sf
import logging
from typing import Optional, Dict, Any, List, Tuple
import sys
import multiprocessing
import concurrent.futures
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

@contextmanager
def temporary_file(suffix: Optional[str] = None):
    """Context manager for temporary file handling"""
    temp_path = tempfile.mktemp(suffix=suffix)
    try:
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")

class ProgressTracker:
    """Tracks progress across multiple processing steps"""
    def __init__(self, status_container, progress_bar):
        self.status = status_container
        self.progress = progress_bar
        self.current_step = 0
        self.total_steps = 5  # Total number of main processing steps
        self.substep_container = st.empty()  # Add container for substep details
        self.metrics_container = st.container()  # Add container for metrics
        
    def update(self, progress: float, message: str, substep: str = "", metrics: Dict[str, Any] = None):
        """Update progress bar and status message with enhanced UI feedback
        
        Args:
            progress: Progress within current step (0-1)
            message: Main status message
            substep: Optional substep detail
            metrics: Optional dictionary of metrics to display
        """
        # Calculate overall progress (each step is 20% of total)
        overall_progress = min((self.current_step + progress) / self.total_steps, 1.0)
        
        # Update progress bar with smoother animation
        self.progress.progress(overall_progress)
        
        # Update main status with color coding
        status_html = f"""
        <div class="status-message {'status-processing' if overall_progress < 1 else 'status-complete'}">
            <h4>{message}</h4>
        """
        if substep:
            status_html += f"<p>{substep}</p>"
        status_html += "</div>"
        
        self.status.markdown(status_html, unsafe_allow_html=True)
        
        # Display metrics if provided
        if metrics:
            with self.metrics_container:
                cols = st.columns(len(metrics))
                for col, (metric_name, metric_value) in zip(cols, metrics.items()):
                    with col:
                        st.metric(
                            label=metric_name,
                            value=metric_value if isinstance(metric_value, (int, float)) else str(metric_value)
                        )
    
    def next_step(self):
        """Move to next processing step with visual feedback"""
        self.current_step = min(self.current_step + 1, self.total_steps)
        
        # Clear substep container for new step
        self.substep_container.empty()
        
        # Update progress with completion animation
        if self.current_step == self.total_steps:
            self.progress.progress(1.0)
            self.status.markdown("""
                <div class="status-message status-complete">
                    <h4>✅ Processing Complete!</h4>
                </div>
            """, unsafe_allow_html=True)
        

    def error(self, message: str):
        """Display error message with visual feedback"""
        self.status.markdown(f"""
            <div class="status-message status-error">
                <h4>❌ Error</h4>
                <p>{message}</p>
            </div>
        """, unsafe_allow_html=True)
        

class AudioFeatureExtractor:
    """Handles audio feature extraction with improved pause detection"""
    def __init__(self):
        self.sr = 16000
        self.hop_length = 512
        self.n_fft = 2048
        self.chunk_duration = 300
        # Parameters for pause detection
        self.min_pause_duration = 4  # minimum pause duration in seconds
        self.silence_threshold = -40    # dB threshold for silence
        
    def _analyze_pauses(self, silent_frames, frame_time):
        """Analyze pauses with minimal memory usage."""
        pause_durations = []
        current_pause = 0

        for is_silent in silent_frames:
            if is_silent:
                current_pause += 1
            elif current_pause > 0:
                duration = current_pause * frame_time
                if duration > 0.5:  # Only count pauses longer than 300ms
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

    def extract_features(self, audio_path: str, progress_callback=None) -> Dict[str, float]:
        try:
            if progress_callback:
                progress_callback(0.1, "Loading audio file...")
            
            # Load audio with proper sample rate
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Calculate amplitude features
            rms = librosa.feature.rms(y=audio)[0]
            mean_amplitude = float(np.mean(rms)) * 100  # Scale for better readability
            
            # Calculate pitch features using pyin
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                sr=sr,
                fmin=70,
                fmax=400,
                frame_length=2048
            )
            
            # Filter out zero and NaN values
            valid_f0 = f0[np.logical_and(voiced_flag == 1, ~np.isnan(f0))]
            
            # Improved pause detection
            S = np.abs(librosa.stft(audio))
            db = librosa.amplitude_to_db(S, ref=np.max)
            frame_length = 2048  # ~128ms at 16kHz
            hop_length = 512     # ~32ms at 16kHz
            
            # Calculate energy in dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            # Detect silence with improved threshold
            silence_threshold = -40
            is_silence = rms_db < silence_threshold
            
            # Count pauses (silence segments longer than 0.5 seconds)
            min_pause_frames = int(0.5 * sr / hop_length)  # 0.5 seconds minimum pause duration
            silence_runs = np.split(is_silence, np.where(np.diff(is_silence))[0] + 1)
            
            # Count valid pauses (longer than minimum duration)
            valid_pauses = sum(1 for run in silence_runs if len(run) >= min_pause_frames and run[0])
            
            # Calculate duration in minutes
            duration_minutes = len(audio) / sr / 60
            
            # Calculate pauses per minute
            pauses_per_minute = valid_pauses / duration_minutes if duration_minutes > 0 else 0
            
            return {
                "pitch_mean": float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0,
                "pitch_std": float(np.std(valid_f0)) if len(valid_f0) > 0 else 0,
                "mean_amplitude": mean_amplitude,
                "amplitude_deviation": float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0,
                "pauses_per_minute": float(pauses_per_minute),
                "duration": float(len(audio) / sr),
                "rising_patterns": int(np.sum(np.diff(valid_f0) > 0)) if len(valid_f0) > 1 else 0,
                "falling_patterns": int(np.sum(np.diff(valid_f0) < 0)) if len(valid_f0) > 1 else 0,
                "variations_per_minute": float(len(valid_f0) / (len(audio) / sr / 60)) if len(audio) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            raise AudioProcessingError(f"Feature extraction failed: {str(e)}")


    def _process_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """Process a single chunk of audio with improved pause detection"""
        # Calculate STFT
        D = librosa.stft(chunk, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.abs(D)
        
        # Calculate RMS energy in dB
        rms = librosa.feature.rms(S=S)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Detect pauses using silence threshold
        is_silence = rms_db < self.silence_threshold
        frame_time = self.hop_length / self.sr
        pause_analysis = self._analyze_pauses(is_silence, frame_time)
        
        # Calculate pitch features
        f0, voiced_flag, _ = librosa.pyin(
            chunk,
            sr=self.sr,
            fmin=70,
            fmax=400,
            frame_length=self.n_fft
        )
        
        return {
            "rms": rms,
            "f0": f0[voiced_flag == 1] if f0 is not None else np.array([]),
            "duration": len(chunk) / self.sr,
            "pause_count": pause_analysis['total_pauses'],
            "mean_pause_duration": pause_analysis['mean_pause_duration']
        }

    def _combine_features(self, features: List[Dict[str, Any]]) -> Dict[str, float]:
        """Combine features from multiple chunks"""
        all_f0 = np.concatenate([f["f0"] for f in features if len(f["f0"]) > 0])
        all_rms = np.concatenate([f["rms"] for f in features])
        
        pitch_mean = np.mean(all_f0) if len(all_f0) > 0 else 0
        pitch_std = np.std(all_f0) if len(all_f0) > 0 else 0
        
        return {
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "mean_amplitude": float(np.mean(all_rms)),
            "amplitude_deviation": float(np.std(all_rms) / np.mean(all_rms)) if np.mean(all_rms) > 0 else 0,
            "rising_patterns": int(np.sum(np.diff(all_f0) > 0)) if len(all_f0) > 1 else 0,
            "falling_patterns": int(np.sum(np.diff(all_f0) < 0)) if len(all_f0) > 1 else 0,
            "variations_per_minute": float((np.sum(np.diff(all_f0) != 0) if len(all_f0) > 1 else 0) / 
                                        (sum(f["duration"] for f in features) / 60))
        }

class ContentAnalyzer:
    """Analyzes teaching content using OpenAI API"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.retry_count = 3
        self.retry_delay = 1
        
    def analyze_content(self, transcript: str, progress_callback=None) -> Dict[str, Any]:
        """Analyze teaching content with strict validation and robust JSON handling"""
        for attempt in range(self.retry_count):
            try:
                if progress_callback:
                    progress_callback(0.2, "Preparing content analysis...")
                
                # Remove any truncation of transcript - pass full text to API
                prompt = self._create_analysis_prompt(transcript)
                logger.info(f"Sending full transcript of length: {len(transcript)} characters")
                
                if progress_callback:
                    progress_callback(0.5, "Processing with AI model...")
                
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": """You are a strict teaching evaluator focusing on core teaching competencies.
                             Maintain high standards and require clear evidence of quality teaching.
                             
                             Score of 1 requires meeting ALL criteria below with clear evidence.
                             Score of 0 if ANY major teaching deficiency is present.
                             
                             Concept Assessment Scoring Criteria:
                             - Subject Matter Accuracy (Score 1 requires: Completely accurate information, no errors)
                             - First Principles Approach (Score 1 requires: Clear explanation of fundamentals before complex topics)
                             - Examples and Business Context (Score 1 requires: At least 2 relevant examples with business context)
                             - Cohesive Storytelling (Score 1 requires: Clear logical flow with smooth transitions)
                             - Engagement and Interaction (Score 1 requires: At least 3 engagement points or questions)
                             - Professional Tone (Score 1 requires: Consistently professional delivery)
                             
                             Code Assessment Scoring Criteria:
                             - Depth of Explanation (Score 1 requires: Thorough explanation of implementation details)
                             - Output Interpretation (Score 1 requires: Clear connection between code outputs and business value)
                             - Breaking down Complexity (Score 1 requires: Systematic breakdown of complex concepts)
                             
                             Major Teaching Deficiencies (ANY of these results in Score 0):
                             - Any factual errors
                             - Missing foundational explanations
                             - Insufficient examples or business context
                             - Disorganized presentation
                             - Limited learner engagement
                             - Unprofessional language
                             - Superficial code explanation
                             - Missing business context
                             - Poor complexity management
                             
                             Citations Requirements:
                             - Include specific timestamps [MM:SS]
                             - Provide examples for both good and poor teaching moments
                             - Note specific instances of criteria being met or missed
                             
                             For each improvement suggestion, categorize it as one of:
                             - COMMUNICATION: Related to speaking, pace, tone, clarity, delivery
                             - TEACHING: Related to explanation, examples, engagement, structure
                             - TECHNICAL: Related to code, implementation, technical concepts
                             
                             Always respond with valid JSON containing these exact categories."""},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.3 # Lower temperature for stricter evaluation
                    )
                    logger.info("API call successful")
                except Exception as api_error:
                    logger.error(f"API call failed: {str(api_error)}")
                    raise
                
                result_text = response.choices[0].message.content.strip()
                logger.info(f"Raw API response: {result_text[:500]}...")
                
                try:
                    result = json.loads(result_text)
                    logger.info("Successfully parsed JSON response")
                    
                    # Validate the response structure
                    required_categories = {
                        "Concept Assessment": [
                            "Subject Matter Accuracy",
                            "First Principles Approach",
                            "Examples and Business Context",
                            "Cohesive Storytelling",
                            "Engagement and Interaction",
                            "Professional Tone"
                        ],
                        "Code Assessment": [
                            "Depth of Explanation",
                            "Output Interpretation",
                            "Breaking down Complexity"
                        ]
                    }
                    
                    # Check if response has required structure
                    for category, subcategories in required_categories.items():
                        if category not in result:
                            logger.error(f"Missing category: {category}")
                            raise ValueError(f"Response missing required category: {category}")
                        
                        for subcategory in subcategories:
                            if subcategory not in result[category]:
                                logger.error(f"Missing subcategory: {subcategory} in {category}")
                                raise ValueError(f"Response missing required subcategory: {subcategory}")
                            
                            subcat_data = result[category][subcategory]
                            if not isinstance(subcat_data, dict):
                                logger.error(f"Invalid format for {category}.{subcategory}")
                                raise ValueError(f"Invalid format for {category}.{subcategory}")
                            
                            if "Score" not in subcat_data or "Citations" not in subcat_data:
                                logger.error(f"Missing Score or Citations in {category}.{subcategory}")
                                raise ValueError(f"Missing Score or Citations in {category}.{subcategory}")
                    
                    return result
                    
                except json.JSONDecodeError as json_error:
                    logger.error(f"JSON parsing error: {str(json_error)}")
                    logger.error(f"Invalid JSON response: {result_text}")
                    raise
                except ValueError as val_error:
                    logger.error(f"Validation error: {str(val_error)}")
                    raise
                
            except Exception as e:
                logger.error(f"Content analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_count - 1:
                    logger.error("All attempts failed, returning default structure")
                    return {
                        "Concept Assessment": {
                            "Subject Matter Accuracy": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "First Principles Approach": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Examples and Business Context": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Cohesive Storytelling": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Engagement and Interaction": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Professional Tone": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]}
                        },
                        "Code Assessment": {
                            "Depth of Explanation": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Output Interpretation": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]},
                            "Breaking down Complexity": {"Score": 0, "Citations": [f"Analysis failed: {str(e)}"]}
                        }
                    }
                time.sleep(self.retry_delay * (2 ** attempt))

    def _create_analysis_prompt(self, transcript: str) -> str:
        """Create the analysis prompt with smart timestamp handling"""
        # First try to extract existing timestamps
        timestamps = re.findall(r'\[(\d{2}:\d{2})\]', transcript)
        
        if timestamps:
            # Use existing timestamps
            timestamp_instruction = f"""Use the EXACT timestamps from the transcript (e.g. {', '.join(timestamps[:3])}).
Do not create new timestamps."""
        else:
            # Calculate approximate timestamps based on word position
            words_per_minute = 150  # average speaking rate
            timestamp_instruction = """Generate timestamps based on word position:
1. Count words from start of transcript
2. Calculate time: (word_count / 150) minutes
3. Format as [MM:SS]
Example: If a quote starts at word 300, timestamp would be [02:00] (300 words / 150 words per minute)"""
            
            # Add word position markers to help with timestamp calculation
            words = transcript.split()
            marked_transcript = ""
            for i, word in enumerate(words):
                if i % 150 == 0:  # Add marker every ~1 minute of speech
                    minutes = i // 150
                    marked_transcript += f"\n[{minutes:02d}:00] "
                marked_transcript += word + " "
            transcript = marked_transcript

        prompt_template = """Analyze this teaching content and provide detailed assessment.

Transcript:
{transcript}

Timestamp Instructions:
{timestamp_instruction}

Required JSON structure:
{{
    "Concept Assessment": {{
        "Subject Matter Accuracy": {{
            "Score": 1,
            "Citations": ["[MM:SS] Quote from transcript"]
        }},
        "First Principles Approach": {{
            "Score": 1,
            "Citations": ["[MM:SS] Quote from transcript"]
        }},
        "Examples and Business Context": {{
            "Score": 1,
            "Citations": ["[MM:SS] Quote from transcript"]
        }},
        "Cohesive Storytelling": {{
            "Score": 1,
            "Citations": ["[MM:SS] Quote from transcript"]
        }},
        "Engagement and Interaction": {{
            "Score": 1,
            "Citations": ["[MM:SS] Quote from transcript"]
        }},
        "Professional Tone": {{
            "Score": 1,
            "Citations": ["[MM:SS] Quote from transcript"]
        }}
    }},
    "Code Assessment": {{
        "Depth of Explanation": {{
            "Score": 1,
            "Citations": ["[MM:SS] Quote from transcript"]
        }},
        "Output Interpretation": {{
            "Score": 1,
            "Citations": ["[MM:SS] Quote from transcript"]
        }},
        "Breaking down Complexity": {{
            "Score": 1,
            "Citations": ["[MM:SS] Quote from transcript"]
        }}
    }}
}}

Evaluation Criteria:
- Subject Matter Accuracy: Check for factual errors or incorrect correlations
- First Principles Approach: Evaluate if fundamentals are explained before technical terms
- Examples and Business Context: Look for real-world examples
- Cohesive Storytelling: Check for logical flow between topics
- Engagement and Interaction: Evaluate use of questions and engagement techniques
- Professional Tone: Assess language and delivery professionalism
- Depth of Explanation: Evaluate technical explanations
- Output Interpretation: Check if code outputs are explained clearly
- Breaking down Complexity: Assess ability to simplify complex concepts

Important:
- Each citation must include a timestamp and relevant quote
- Citations should highlight specific examples of criteria being met or missed
- Use only Score values of 0 or 1"""

        return prompt_template.format(
            transcript=transcript,
            timestamp_instruction=timestamp_instruction
        )

    def _evaluate_speech_metrics(self, transcript: str, audio_features: Dict[str, float], 
                           progress_callback=None) -> Dict[str, Any]:
        """Evaluate speech metrics with improved accuracy"""
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating speech metrics...")

            # Calculate words and duration
            words = len(transcript.split())
            duration_minutes = float(audio_features.get('duration', 0)) / 60
            
            # Calculate words per minute with updated range (130-160 WPM is ideal for teaching)
            words_per_minute = float(words / duration_minutes if duration_minutes > 0 else 0)
            
            # Improved filler word detection (2-3 per minute is acceptable)
            filler_words = re.findall(r'\b(um|uh|like|you\s+know|basically|actually|literally)\b', 
                                    transcript.lower())
            fillers_count = len(filler_words)
            fillers_per_minute = float(fillers_count / duration_minutes if duration_minutes > 0 else 0)
            
            # Improved error detection (1-2 per minute is acceptable)
            repeated_words = len(re.findall(r'\b(\w+)\s+\1\b', transcript.lower()))
            incomplete_sentences = len(re.findall(r'[a-zA-Z]+\s*\.\.\.|\b[a-zA-Z]+\s*-\s+', transcript))
            errors_count = repeated_words + incomplete_sentences
            errors_per_minute = float(errors_count / duration_minutes if duration_minutes > 0 else 0)
            
            # Ensure all values are properly cast to float and have fallbacks
            pitch_mean = float(audio_features.get("pitch_mean", 0))
            pitch_std = float(audio_features.get("pitch_std", 0))
            mean_amplitude = float(audio_features.get("mean_amplitude", 0))
            amplitude_deviation = float(audio_features.get("amplitude_deviation", 0))
            pauses_per_minute = float(audio_features.get("pauses_per_minute", 0))
            variations_per_minute = float(audio_features.get("variations_per_minute", 0))
            rising_patterns = int(audio_features.get("rising_patterns", 0))
            falling_patterns = int(audio_features.get("falling_patterns", 0))
            
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 180 else 0,
                    "wpm": words_per_minute,
                    "total_words": words,
                    "duration_minutes": duration_minutes
                },
                "fluency": {
                    "score": 1 if errors_per_minute <= 2 and fillers_per_minute <= 3 else 0,
                    "fillersPerMin": fillers_per_minute,
                    "errorsPerMin": errors_per_minute
                },
                "flow": {
                    "score": 1 if pauses_per_minute <= 12 else 0,
                    "pausesPerMin": pauses_per_minute
                },
                "intonation": {
                    "pitch": pitch_mean,
                    "pitchScore": 1 if 20 <= (pitch_std / pitch_mean * 100 if pitch_mean > 0 else 0) <= 40 else 0,
                    "pitchVariation": pitch_std,
                    "patternScore": 1 if variations_per_minute >= 100 else 0,
                    "risingPatterns": rising_patterns,
                    "fallingPatterns": falling_patterns,
                    "variationsPerMin": variations_per_minute,
                    "mu": pitch_mean
                },
                "energy": {
                    "score": 1 if 60 <= mean_amplitude <= 75 else 0,
                    "meanAmplitude": mean_amplitude,
                    "amplitudeDeviation": amplitude_deviation,
                    "variationScore": 1 if 0.05 <= amplitude_deviation <= 0.15 else 0
                }
            }

        except Exception as e:
            logger.error(f"Error in speech metrics evaluation: {e}")
            raise

    def generate_suggestions(self, category: str, citations: List[str]) -> List[str]:
        """Generate contextual suggestions based on category and citations"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a teaching expert providing specific, actionable suggestions 
                    for improvement. Focus on the single most important, practical advice based on the teaching category 
                    and cited issues. Keep suggestions under 25 words."""},
                    {"role": "user", "content": f"""
                    Teaching Category: {category}
                    Issues identified in citations:
                    {json.dumps(citations, indent=2)}
                    
                    Please provide 2 or 3 at max specific, actionable suggestion for improvement.
                    Format as a JSON array with a single string."""}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("suggestions", [])
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [f"Unable to generate specific suggestions: {str(e)}"]

class RecommendationGenerator:
    """Generates teaching recommendations using OpenAI API"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.retry_count = 3
        self.retry_delay = 1
        
    def generate_recommendations(self, 
                           metrics: Dict[str, Any], 
                           content_analysis: Dict[str, Any], 
                           progress_callback=None) -> Dict[str, Any]:
        """Generate recommendations with robust JSON handling"""
        for attempt in range(self.retry_count):
            try:
                if progress_callback:
                    progress_callback(0.2, "Preparing recommendation analysis...")
                
                prompt = self._create_recommendation_prompt(metrics, content_analysis)
                
                if progress_callback:
                    progress_callback(0.5, "Generating recommendations...")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": """You are a teaching expert providing actionable recommendations. 
                        Each improvement must be categorized as one of:
                        - COMMUNICATION: Related to speaking, pace, tone, clarity, delivery
                        - TEACHING: Related to explanation, examples, engagement, structure
                        - TECHNICAL: Related to code, implementation, technical concepts
                        
                        Always respond with a valid JSON object containing categorized improvements."""},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                if progress_callback:
                    progress_callback(0.8, "Formatting recommendations...")
                
                result_text = response.choices[0].message.content.strip()
                
                try:
                    result = json.loads(result_text)
                    # Ensure improvements are properly formatted
                    if "improvements" in result:
                        formatted_improvements = []
                        for imp in result["improvements"]:
                            if isinstance(imp, str):
                                # Default categorization for legacy format
                                formatted_improvements.append({
                                    "category": "TECHNICAL",
                                    "message": imp
                                })
                            elif isinstance(imp, dict):
                                # Ensure proper structure for dict format
                                formatted_improvements.append({
                                    "category": imp.get("category", "TECHNICAL"),
                                    "message": imp.get("message", str(imp))
                                })
                        result["improvements"] = formatted_improvements
                except json.JSONDecodeError:
                    result = {
                        "geographyFit": "Unknown",
                        "improvements": [
                            {
                                "category": "TECHNICAL",
                                "message": "Unable to generate specific recommendations"
                            }
                        ],
                        "rigor": "Undetermined",
                        "profileMatches": []
                    }
                
                if progress_callback:
                    progress_callback(1.0, "Recommendations complete!")
                
                return result
                
            except Exception as e:
                logger.error(f"Recommendation generation attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_count - 1:
                    return {
                        "geographyFit": "Unknown",
                        "improvements": [
                            {
                                "category": "TECHNICAL",
                                "message": f"Unable to generate specific recommendations: {str(e)}"
                            }
                        ],
                        "rigor": "Undetermined",
                        "profileMatches": []
                    }
                time.sleep(self.retry_delay * (2 ** attempt))
    
    def _create_recommendation_prompt(self, metrics: Dict[str, Any], content_analysis: Dict[str, Any]) -> str:
        """Create the recommendation prompt"""
        return f"""Based on the following metrics and analysis, provide recommendations:
Metrics: {json.dumps(metrics)}
Content Analysis: {json.dumps(content_analysis)}

Analyze the teaching style and provide:
1. A concise performance summary (2-3 paragraphs highlighting key strengths and areas for improvement)
2. Geography fit assessment
3. Specific improvements needed (each must be categorized as COMMUNICATION, TEACHING, or TECHNICAL)
4. Profile matching for different learner types (choose ONLY ONE best match)
5. Overall teaching rigor assessment

Required JSON structure:
{{
    "summary": "Comprehensive summary of teaching performance, strengths, and areas for improvement",
    "geographyFit": "String describing geographical market fit",
    "improvements": [
        {{
            "category": "COMMUNICATION",
            "message": "Specific improvement recommendation"
        }},
        {{
            "category": "TEACHING",
            "message": "Specific improvement recommendation"
        }},
        {{
            "category": "TECHNICAL",
            "message": "Specific improvement recommendation"
        }}
    ],
    "rigor": "Assessment of teaching rigor",
    "profileMatches": [
        {{
            "profile": "junior_technical",
            "match": false,
            "reason": "Detailed explanation why this profile is not the best match"
        }},
        {{
            "profile": "senior_non_technical",
            "match": false,
            "reason": "Detailed explanation why this profile is not the best match"
        }},
        {{
            "profile": "junior_expert",
            "match": false,
            "reason": "Detailed explanation why this profile is not the best match"
        }},
        {{
            "profile": "senior_expert",
            "match": false,
            "reason": "Detailed explanation why this profile is not the best match"
        }}
    ]
}}

Consider:
- Teaching pace and complexity level
- Balance of technical vs business context
- Depth of code explanations
- Use of examples and analogies
- Engagement style
- Communication metrics
- Teaching assessment scores"""

class CostCalculator:
    """Calculates API and processing costs"""
    def __init__(self):
        self.GPT4_INPUT_COST = 0.15 / 1_000_000  # $0.15 per 1M tokens input
        self.GPT4_OUTPUT_COST = 0.60 / 1_000_000  # $0.60 per 1M tokens output
        self.WHISPER_COST = 0.006 / 60  # $0.006 per minute
        self.costs = {
            'transcription': 0.0,
            'content_analysis': 0.0,
            'recommendations': 0.0,
            'total': 0.0
        }

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count based on words"""
        return len(text.split()) * 1.3  # Approximate tokens per word

    def add_transcription_cost(self, duration_seconds: float):
        """Calculate Whisper transcription cost"""
        cost = (duration_seconds / 60) * self.WHISPER_COST
        self.costs['transcription'] = cost
        self.costs['total'] += cost
        print(f"\nTranscription Cost: ${cost:.4f}")

    def add_gpt4_cost(self, input_text: str, output_text: str, operation: str):
        """Calculate GPT-4 API cost for a single operation"""
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        input_cost = input_tokens * self.GPT4_INPUT_COST
        output_cost = output_tokens * self.GPT4_OUTPUT_COST
        total_cost = input_cost + output_cost
        
        self.costs[operation] = total_cost
        self.costs['total'] += total_cost
        
        print(f"\n{operation.replace('_', ' ').title()} Cost:")
        print(f"Input tokens: {input_tokens:.0f} (${input_cost:.4f})")
        print(f"Output tokens: {output_tokens:.0f} (${output_cost:.4f})")
        print(f"Operation total: ${total_cost:.4f}")

    def print_total_cost(self):
        """Print total cost breakdown"""
        print("\n=== Cost Breakdown ===")
        for key, cost in self.costs.items():
            if key != 'total':
                print(f"{key.replace('_', ' ').title()}: ${cost:.4f}")
        print(f"\nTotal Cost: ${self.costs['total']:.4f}")

class MentorEvaluator:
    """Main class for video evaluation"""
    def __init__(self, model_cache_dir: Optional[str] = None):
        # Fix potential API key issue
        self.api_key = st.secrets.get("OPENAI_API_KEY")  # Use get() method
        if not self.api_key:
            raise ValueError("OpenAI API key not found in secrets")
        
        # Add error handling for model cache directory
        try:
            if model_cache_dir:
                self.model_cache_dir = Path(model_cache_dir)
            else:
                self.model_cache_dir = Path.home() / ".cache" / "whisper"
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create model cache directory: {e}")
            
        # Initialize components with proper error handling
        try:
            self.feature_extractor = AudioFeatureExtractor()
            self.content_analyzer = ContentAnalyzer(self.api_key)
            self.recommendation_generator = RecommendationGenerator(self.api_key)
            self.cost_calculator = CostCalculator()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {e}")

    def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return value
        return None

    def _set_cached_result(self, key: str, value: Any):
        """Cache result with timestamp"""
        self._cache[key] = (time.time(), value)

    def _extract_audio(self, video_path: str, output_path: str, progress_callback=None) -> str:
        """Extract audio from video"""
        try:
            if progress_callback:
                progress_callback(0.1, "Checking dependencies...")

            if not shutil.which('ffmpeg'):
                raise AudioProcessingError("FFmpeg is not installed")

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            if progress_callback:
                progress_callback(0.3, "Configuring audio extraction...")

            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ar', '16000',  # Set sample rate to 16kHz
                '-ac', '1',      # Convert to mono
                '-f', 'wav',     # Output format
                '-v', 'warning', # Reduce verbosity
                '-y',           # Overwrite output file
                output_path
            ]

            if progress_callback:
                progress_callback(0.5, "Extracting audio...")

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise AudioProcessingError(f"FFmpeg Error: {result.stderr}")

            if not os.path.exists(output_path):
                raise AudioProcessingError("Audio extraction failed: output file not created")

            if progress_callback:
                progress_callback(1.0, "Audio extraction complete!")

            return output_path

        except Exception as e:
            logger.error(f"Error in audio extraction: {e}")
            raise AudioProcessingError(f"Audio extraction failed: {str(e)}")

    def _preprocess_audio(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Preprocess audio for analysis"""
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input audio file not found: {input_path}")

            # If no output path specified, use the input path
            if output_path is None:
                output_path = input_path

            # Load audio
            audio, sr = librosa.load(input_path, sr=16000)

            # Apply preprocessing steps
            # 1. Normalize audio
            audio = librosa.util.normalize(audio)

            # 2. Remove silence
            non_silent = librosa.effects.trim(audio, top_db=20)[0]

            # 3. Save processed audio
            sf.write(output_path, non_silent, sr)

            return output_path

        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            raise AudioProcessingError(f"Audio preprocessing failed: {str(e)}")

    def evaluate_video(self, video_path: str, transcript_file: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Add input validation
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Validate video file format
            valid_extensions = {'.mp4', '.avi', '.mov'}
            if not any(video_path.lower().endswith(ext) for ext in valid_extensions):
                raise ValueError("Unsupported video format. Use MP4, AVI, or MOV")

            # Create progress tracking containers with error handling
            try:
                status = st.empty()
                progress = st.progress(0.0)
                tracker = ProgressTracker(status, progress)
            except Exception as e:
                logger.error(f"Failed to create progress trackers: {e}")
                raise

            # Add cleanup for temporary files
            temp_files = []
            try:
                with temporary_file(suffix=".wav") as temp_audio, \
                     temporary_file(suffix=".wav") as processed_audio:
                    temp_files.extend([temp_audio, processed_audio])
                    
                    # Step 1: Extract audio from video
                    tracker.update(0.1, "Extracting audio from video")
                    self._extract_audio(video_path, temp_audio)
                    tracker.next_step()
                    
                    # Step 2: Preprocess audio
                    tracker.update(0.2, "Preprocessing audio")
                    self._preprocess_audio(temp_audio, processed_audio)
                    tracker.next_step()
                    
                    # Step 3: Extract features
                    tracker.update(0.4, "Extracting audio features")
                    audio_features = self.feature_extractor.extract_features(processed_audio)
                    tracker.next_step()
                    
                    # Step 4: Get transcript - Modified to handle 3-argument progress callback
                    tracker.update(0.6, "Processing transcript")
                    if transcript_file:
                        transcript = transcript_file.getvalue().decode('utf-8')
                    else:
                        # Update progress callback to handle 3 arguments
                        tracker.update(0.6, "Transcribing audio")
                        transcript = self._transcribe_audio(
                            processed_audio, 
                            lambda p, m, extra=None: tracker.update(0.6 + p * 0.2, m)
                        )
                    tracker.next_step()
                    
                    # Step 5: Analyze content
                    tracker.update(0.8, "Analyzing teaching content")
                    content_analysis = self.content_analyzer.analyze_content(transcript)
                    
                    # Step 6: Generate recommendations
                    tracker.update(0.9, "Generating recommendations")
                    recommendations = self.recommendation_generator.generate_recommendations(
                        audio_features,
                        content_analysis
                    )
                    tracker.next_step()
                    
                    # Clear progress indicators
                    status.empty()
                    progress.empty()
                    
                    return {
                        "audio_features": audio_features,
                        "transcript": transcript,
                        "teaching": content_analysis,
                        "recommendations": recommendations
                    }

            finally:
                # Clean up any remaining temporary files
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {temp_file}: {e}")

        except Exception as e:
            logger.error(f"Error in video evaluation: {e}")
            # Clean up UI elements on error
            if 'status' in locals():
                status.empty()
            if 'progress' in locals():
                progress.empty()
            raise RuntimeError(f"Analysis failed: {str(e)}")

    def _transcribe_audio(self, audio_path: str, progress_callback=None) -> str:
        """Transcribe audio with optimized segment detection and detailed progress tracking"""
        try:
            if progress_callback:
                progress_callback(0.1, "Loading transcription model...")

            # Check if GPU is available and set device accordingly
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            # Generate cache key based on file content
            cache_key = f"transcript_{hashlib.md5(open(audio_path, 'rb').read()).hexdigest()}"
            
            # Check cache first
            if cache_key in st.session_state:
                logger.info("Using cached transcription") 
                if progress_callback:
                    progress_callback(1.0, "Retrieved from cache")
                return st.session_state[cache_key]

            # Add validation for audio file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            try:
                audio_info = sf.info(audio_path)
                if audio_info.samplerate != 16000:
                    logger.warning(f"Audio sample rate is {audio_info.samplerate}Hz, expected 16000Hz")
            except Exception as e:
                logger.error(f"Error checking audio file: {e}")
                raise ValueError(f"Invalid audio file: {str(e)}")

            if progress_callback:
                progress_callback(0.2, "Initializing model...")

            # Initialize model with optimized settings and proper error handling
            try:
                model = WhisperModel(
                    "small",
                    device=device,
                    compute_type=compute_type,
                    download_root=self.model_cache_dir,
                    local_files_only=False,
                    cpu_threads=4,
                    num_workers=2
                )
            except Exception as e:
                logger.error(f"Error initializing Whisper model: {e}")
                raise RuntimeError(f"Failed to initialize transcription model: {str(e)}")

            if progress_callback:
                progress_callback(0.3, "Starting transcription...")

            # Get audio duration for progress calculation
            total_duration = audio_info.duration

            # Transcribe with optimized VAD settings and error handling
            try:
                segments, _ = model.transcribe(
                    audio_path,
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=100,
                        threshold=0.3,
                        min_speech_duration_ms=250
                    ),
                    language='en'
                )
            except Exception as e:
                logger.error(f"Error during transcription: {e}")
                raise RuntimeError(f"Transcription failed: {str(e)}")

            # Process segments with better error handling and validation
            transcript_parts = []
            segments = list(segments)  # Convert generator to list
            total_segments = len(segments)
            batch_size = 10
            
            if total_segments == 0:
                logger.warning("No speech segments detected")
                raise ValueError("No speech detected in audio file")

            for i, segment in enumerate(segments, 1):
                if segment.text:  # Only add non-empty segments
                    # Validate segment text
                    cleaned_text = segment.text.strip()
                    if cleaned_text:
                        transcript_parts.append(cleaned_text)
                
                # Update progress less frequently for better performance
                if i % 5 == 0 or i == total_segments:
                    progress = min(i / total_segments, 1.0)
                    progress = 0.3 + (progress * 0.6)
                    
                    current_batch = (i - 1) // batch_size + 1
                    total_batches = (total_segments + batch_size - 1) // batch_size

                    if progress_callback:
                        progress_callback(
                            progress,
                            f"Transcribing Batch {current_batch}/{total_batches}",
                            f"Processing segment {i} of {total_segments}"
                        )

            # Validate final transcript
            transcript = ' '.join(transcript_parts)
            if not transcript.strip():
                raise ValueError("Transcription produced empty result")

            # Cache the result
            st.session_state[cache_key] = transcript

            if progress_callback:
                progress_callback(1.0, "Transcription complete!")

            return transcript

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            if progress_callback:
                progress_callback(1.0, "Error in transcription", str(e))
            raise

    def _merge_transcripts(self, transcripts: List[str]) -> str:
        """Merge transcripts with overlap deduplication"""
        if not transcripts:
            return ""
        
        def clean_text(text):
            # Remove extra spaces and normalize punctuation
            return ' '.join(text.split())
        
        def find_overlap(text1, text2):
            # Find overlapping text between consecutive chunks
            words1 = text1.split()
            words2 = text2.split()
            
            for i in range(min(len(words1), 20), 0, -1):  # Check up to 20 words
                if ' '.join(words1[-i:]) == ' '.join(words2[:i]):
                    return i
            return 0

        merged = clean_text(transcripts[0])
        
        for i in range(1, len(transcripts)):
            current = clean_text(transcripts[i])
            overlap_size = find_overlap(merged, current)
            merged += ' ' + current.split(' ', overlap_size)[-1]
        
        return merged

    def calculate_speech_metrics(self, transcript: str, audio_duration: float) -> Dict[str, float]:
        """Calculate words per minute and other speech metrics."""
        words = len(transcript.split())
        minutes = audio_duration / 60
        return {
            'words_per_minute': words / minutes if minutes > 0 else 0,
            'total_words': words,
            'duration_minutes': minutes
        }

    def _evaluate_speech_metrics(self, transcript: str, audio_features: Dict[str, float], 
                               progress_callback=None) -> Dict[str, Any]:
        """Evaluate speech metrics with improved accuracy"""
        try:
            if progress_callback:
                progress_callback(0.2, "Calculating speech metrics...")

            # Calculate words and duration
            words = len(transcript.split())
            duration_minutes = float(audio_features.get('duration', 0)) / 60
            
            # Calculate words per minute with updated range (130-160 WPM is ideal for teaching)
            words_per_minute = float(words / duration_minutes if duration_minutes > 0 else 0)
            
            # Improved filler word detection (2-3 per minute is acceptable)
            filler_words = re.findall(r'\b(um|uh|like|you\s+know|basically|actually|literally)\b', 
                                    transcript.lower())
            fillers_count = len(filler_words)
            fillers_per_minute = float(fillers_count / duration_minutes if duration_minutes > 0 else 0)
            
            # Improved error detection (1-2 per minute is acceptable)
            repeated_words = len(re.findall(r'\b(\w+)\s+\1\b', transcript.lower()))
            incomplete_sentences = len(re.findall(r'[a-zA-Z]+\s*\.\.\.|\b[a-zA-Z]+\s*-\s+', transcript))
            errors_count = repeated_words + incomplete_sentences
            errors_per_minute = float(errors_count / duration_minutes if duration_minutes > 0 else 0)
            
            # Ensure all values are properly cast to float and have fallbacks
            pitch_mean = float(audio_features.get("pitch_mean", 0))
            pitch_std = float(audio_features.get("pitch_std", 0))
            mean_amplitude = float(audio_features.get("mean_amplitude", 0))
            amplitude_deviation = float(audio_features.get("amplitude_deviation", 0))
            pauses_per_minute = float(audio_features.get("pauses_per_minute", 0))
            variations_per_minute = float(audio_features.get("variations_per_minute", 0))
            rising_patterns = int(audio_features.get("rising_patterns", 0))
            falling_patterns = int(audio_features.get("falling_patterns", 0))
            
            return {
                "speed": {
                    "score": 1 if 120 <= words_per_minute <= 180 else 0,
                    "wpm": words_per_minute,
                    "total_words": words,
                    "duration_minutes": duration_minutes
                },
                "fluency": {
                    "score": 1 if errors_per_minute <= 2 and fillers_per_minute <= 3 else 0,
                    "fillersPerMin": fillers_per_minute,
                    "errorsPerMin": errors_per_minute
                },
                "flow": {
                    "score": 1 if pauses_per_minute <= 12 else 0,
                    "pausesPerMin": pauses_per_minute
                },
                "intonation": {
                    "pitch": pitch_mean,
                    "pitchScore": 1 if 20 <= (pitch_std / pitch_mean * 100 if pitch_mean > 0 else 0) <= 40 else 0,
                    "pitchVariation": pitch_std,
                    "patternScore": 1 if variations_per_minute >= 100 else 0,
                    "risingPatterns": rising_patterns,
                    "fallingPatterns": falling_patterns,
                    "variationsPerMin": variations_per_minute,
                    "mu": pitch_mean
                },
                "energy": {
                    "score": 1 if 60 <= mean_amplitude <= 75 else 0,
                    "meanAmplitude": mean_amplitude,
                    "amplitudeDeviation": amplitude_deviation,
                    "variationScore": 1 if 0.05 <= amplitude_deviation <= 0.15 else 0
                }
            }

        except Exception as e:
            logger.error(f"Error in speech metrics evaluation: {e}")
            raise

def validate_video_file(file_path: str):
    """Validate video file before processing"""
    valid_extensions = {'.mp4', '.avi', '.mov'}
    
    if not os.path.exists(file_path):
        raise ValueError("Video file does not exist")
        
    if os.path.splitext(file_path)[1].lower() not in valid_extensions:
        raise ValueError("Unsupported video format")
        
    if os.path.getsize(file_path) > 1 * 1024 * 1024 * 1024:  # 1GB
        raise ValueError("File size exceeds 1GB limit")
        
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', file_path],
            capture_output=True,
            text=True
        )
        if probe.returncode != 0:
            raise ValueError("Invalid video file")
    except subprocess.SubprocessError:
        raise ValueError("Unable to validate video file")

def display_evaluation(evaluation: Dict[str, Any]):
    """Display evaluation results with improved metrics visualization"""
    try:
        tabs = st.tabs(["Communication", "Teaching", "Recommendations", "Transcript"])
        
        with tabs[0]:
            st.header("Communication Metrics")
            
            # Get audio features directly from evaluation data
            audio_features = evaluation.get("audio_features", {})
            
            # Speed Metrics
            with st.expander("🏃 Speed", expanded=True):
                duration_minutes = float(audio_features.get('duration', 0)) / 60
                words = len(evaluation.get("transcript", "").split())
                wpm = float(words / duration_minutes if duration_minutes > 0 else 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "✅ Pass" if 120 <= wpm <= 180 else "❌ Needs Improvement")
                    st.metric("Words per Minute", f"{wpm:.1f}")
                with col2:
                    st.info("""
                    **Acceptable Range:** 120-180 WPM
                    """)
                    
                    # Add explanation card
                    st.markdown("""
                    <div class="metric-explanation-card">
                        <h4>🎯 Understanding Speed Metrics</h4>
                        <ul>
                            <li><b>Words per Minute (WPM):</b> Rate of speech delivery
                                <br>• Too slow (<120 WPM): May lose audience engagement
                                <br>• Too fast (>180 WPM): May hinder comprehension
                                <br>• Optimal (130-160 WPM): Best for learning</li>
                            <li><b>Total Words:</b> Complete word count in the presentation</li>
                            <li><b>Duration:</b> Total speaking time in minutes</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Fluency Metrics
            with st.expander("🗣️ Fluency", expanded=True):
                # Calculate fluency metrics from audio features
                pitch_mean = float(audio_features.get("pitch_mean", 0))
                pitch_std = float(audio_features.get("pitch_std", 0))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pitch Mean", f"{pitch_mean:.1f} Hz")
                    st.metric("Pitch Variation", f"{pitch_std:.1f} Hz")
                with col2:
                    st.info("""
                    **Acceptable Ranges:**
                    - Pitch Variation: 20-40% from baseline
                    - Variations per Minute: >100
                    """)
                    
                    # Add new explanation card
                    st.markdown("""
                    <div class="metric-explanation-card">
                        <h4>📊 Understanding Intonation Metrics</h4>
                        <ul>
                            <li><b>Pitch Mean (μ):</b> Average voice frequency. Typical ranges:
                                <br>• Male: 85-180 Hz
                                <br>• Female: 165-255 Hz</li>
                            <li><b>Pitch Variation (σ):</b> How much your pitch changes. Higher values indicate more dynamic speech.</li>
                            <li><b>Rising Patterns:</b> Number of upward pitch changes, often used for questions or emphasis.</li>
                            <li><b>Falling Patterns:</b> Number of downward pitch changes, typically used for statements or conclusions.</li>
                            <li><b>Variations per Minute:</b> Total pitch changes per minute. Higher values indicate more engaging speech patterns.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Flow Metrics
            with st.expander("🌊 Flow", expanded=True):
                pauses_per_minute = float(audio_features.get("pauses_per_minute", 0))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "✅ Pass" if pauses_per_minute <= 12 else "❌ Needs Improvement")
                    st.metric("Pauses per Minute", f"{pauses_per_minute:.1f}")
                with col2:
                    st.info("**Acceptable Range:** < 12 PPM")
                    
                    st.markdown("""
                    <div class="metric-explanation-card">
                        <h4>🌊 Understanding Flow Metrics</h4>
                        <ul>
                            <li><b>Pauses per Minute (PPM):</b> Frequency of speech breaks
                                <br>• Strategic pauses (8-12 PPM): Aid comprehension
                                <br>• Too few: May sound rushed
                                <br>• Too many: Can disrupt flow</li>
                            <li><b>Pause Duration:</b> Length of speech breaks
                                <br>• Short pauses (0.5-1s): Natural rhythm
                                <br>• Long pauses (>2s): Should be intentional</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Intonation Metrics
            with st.expander("🎵 Intonation", expanded=True):
                variations_per_minute = float(audio_features.get("variations_per_minute", 0))
                rising_patterns = int(audio_features.get("rising_patterns", 0))
                falling_patterns = int(audio_features.get("falling_patterns", 0))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Variations per Minute", f"{variations_per_minute:.1f}")
                    st.metric("Rising Patterns", rising_patterns)
                    st.metric("Falling Patterns", falling_patterns)
                with col2:
                    st.info("""
                    **Acceptable Ranges:**
                    - Variations per Minute: >100
                    - Rising Patterns: Number of upward pitch changes
                    - Falling Patterns: Number of downward pitch changes
                    """)
                    
                    # Add new explanation card
                    st.markdown("""
                    <div class="metric-explanation-card">
                        <h4>📊 Understanding Intonation Metrics</h4>
                        <ul>
                            <li><b>Pitch Mean (μ):</b> Average voice frequency. Typical ranges:
                                <br>• Male: 85-180 Hz
                                <br>• Female: 165-255 Hz</li>
                            <li><b>Pitch Variation (σ):</b> How much your pitch changes. Higher values indicate more dynamic speech.</li>
                            <li><b>Rising Patterns:</b> Number of upward pitch changes, often used for questions or emphasis.</li>
                            <li><b>Falling Patterns:</b> Number of downward pitch changes, typically used for statements or conclusions.</li>
                            <li><b>Variations per Minute:</b> Total pitch changes per minute. Higher values indicate more engaging speech patterns.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            # Energy Metrics
            with st.expander("⚡ Energy", expanded=True):
                mean_amplitude = float(audio_features.get("mean_amplitude", 0))
                amplitude_deviation = float(audio_features.get("amplitude_deviation", 0))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", "✅ Pass" if 60 <= mean_amplitude <= 75 else "❌ Needs Improvement")
                    st.metric("Mean Amplitude", f"{mean_amplitude:.1f}")
                    st.metric("Amplitude Deviation", f"{amplitude_deviation:.2f}")
                with col2:
                    st.info("""
                    **Acceptable Ranges:**
                    - Mean Amplitude: 65-85 dB
                    - Amplitude Deviation: 0.15-0.35
                    """)
                    
                    st.markdown("""
                    <div class="metric-explanation-card">
                        <h4>⚡ Understanding Energy Metrics</h4>
                        <ul>
                            <li><b>Mean Amplitude:</b> Average voice volume
                                <br>• Below 65 dB: Too quiet for classroom
                                <br>• 65-85 dB: Optimal teaching range
                                <br>• Above 85 dB: May cause listener fatigue</li>
                            <li><b>Amplitude Deviation:</b> Voice volume variation
                                <br>• Below 0.15: Too monotone
                                <br>• 0.15-0.35: Natural variation
                                <br>• Above 0.35: Excessive variation</li>
                            <li><b>Variation Score:</b> Overall energy dynamics
                                <br>• Measures consistency of voice projection
                                <br>• Reflects engagement and emphasis</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

        with tabs[1]:
            st.header("Teaching Analysis")
            
            teaching_data = evaluation.get("teaching", {})
            content_analyzer = ContentAnalyzer(st.secrets["OPENAI_API_KEY"])
            
            # Display Concept Assessment with AI-generated suggestions
            with st.expander("📚 Concept Assessment", expanded=True):
                concept_data = teaching_data.get("Concept Assessment", {})
                
                for category, details in concept_data.items():
                    score = details.get("Score", 0)
                    citations = details.get("Citations", [])
                    
                    # Get AI-generated suggestions if score is 0
                    suggestions = []
                    if score == 0:
                        suggestions = content_analyzer.generate_suggestions(category, citations)
                    
                    # Create suggestions based on score and category
                    st.markdown(f"""
                        <div class="teaching-card">
                            <div class="teaching-header">
                                <span class="category-name">{category}</span>
                                <span class="score-badge {'score-pass' if score == 1 else 'score-fail'}">
                                    {'✅ Pass' if score == 1 else '❌ Needs Work'}
                                </span>
                            </div>
                            <div class="citations-container">
                    """, unsafe_allow_html=True)
                    
                    # Display citations
                    for citation in citations:
                        st.markdown(f"""
                            <div class="citation-box">
                                <i class="citation-text">{citation}</i>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Display AI-generated suggestions if score is 0
                    if score == 0 and suggestions:
                        st.markdown("""
                            <div class="suggestions-box">
                                <h4>🎯 Suggestions for Improvement:</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        for suggestion in suggestions:
                            st.markdown(f"""
                                <div class="suggestion-item">
                                    • {suggestion}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    st.markdown("---")
            
            # Display Code Assessment with AI-generated suggestions
            with st.expander("💻 Code Assessment", expanded=True):
                code_data = teaching_data.get("Code Assessment", {})
                
                for category, details in code_data.items():
                    score = details.get("Score", 0)
                    citations = details.get("Citations", [])
                    
                    # Get AI-generated suggestions if score is 0
                    suggestions = []
                    if score == 0:
                        suggestions = content_analyzer.generate_suggestions(category, citations)
                    
                    # Create suggestions based on score and category
                    st.markdown(f"""
                        <div class="teaching-card">
                            <div class="teaching-header">
                                <span class="category-name">{category}</span>
                                <span class="score-badge {'score-pass' if score == 1 else 'score-fail'}">
                                    {'✅ Pass' if score == 1 else '❌ Needs Work'}
                                </span>
                            </div>
                            <div class="citations-container">
                    """, unsafe_allow_html=True)
                    
                    for citation in citations:
                        st.markdown(f"""
                            <div class="citation-box">
                                <i class="citation-text">{citation}</i>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Display AI-generated suggestions if score is 0
                    if score == 0 and suggestions:
                        st.markdown("""
                            <div class="suggestions-box">
                                <h4>🎯Suggestions for Improvement:</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        for suggestion in suggestions:
                            st.markdown(f"""
                                <div class="suggestion-item">
                                    • {suggestion}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    st.markdown("---")

        with tabs[2]:
            st.header("Recommendations")
            recommendations = evaluation.get("recommendations", {})
            
            # Display summary in a styled card
            if "summary" in recommendations:
                st.markdown("""
                    <div class="summary-card">
                        <h4>📊 Overall Summary</h4>
                        <div class="summary-content">
                """, unsafe_allow_html=True)
                st.markdown(recommendations["summary"])
                st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Display improvements using categories from content analysis
            st.markdown("<h4>💡 Areas for Improvement</h4>", unsafe_allow_html=True)
            improvements = recommendations.get("improvements", [])
            
            if isinstance(improvements, list):
                # Use predefined categories
                categories = {
                    "🗣️ Communication": [],
                    "📚 Teaching": [],
                    "💻 Technical": []
                }
                
                # Each improvement should now come with a category from the content analysis
                for improvement in improvements:
                    if isinstance(improvement, dict):
                        category = improvement.get("category", "💻 Technical")  # Default to Technical if no category
                        message = improvement.get("message", str(improvement))
                        if "COMMUNICATION" in category.upper():
                            categories["🗣️ Communication"].append(message)
                        elif "TEACHING" in category.upper():
                            categories["📚 Teaching"].append(message)
                        elif "TECHNICAL" in category.upper():
                            categories["💻 Technical"].append(message)
                    else:
                        # Handle legacy format or plain strings
                        categories["💻 Technical"].append(improvement)
                
                # Display categorized improvements in columns
                cols = st.columns(len(categories))
                for col, (category, items) in zip(cols, categories.items()):
                    with col:
                        st.markdown(f"""
                            <div class="improvement-card">
                                <h5>{category}</h5>
                                <div class="improvement-list">
                        """, unsafe_allow_html=True)
                        
                        for item in items:
                            st.markdown(f"""
                                <div class="improvement-item">
                                    • {item}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Add additional CSS for new components
            st.markdown("""
                <style>
                .teaching-card {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .teaching-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                }
                
                .category-name {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #1f77b4;
                }
                
                .score-badge {
                    padding: 5px 15px;
                    border-radius: 15px;
                    font-weight: bold;
                }
                
                .score-pass {
                    background-color: #28a745;
                    color: white;
                }
                
                .score-fail {
                    background-color: #dc3545;
                    color: white;
                }
                
                .citations-container {
                    margin-top: 10px;
                }
                
                .citation-box {
                    background: #f8f9fa;
                    border-left: 3px solid #6c757d;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 0 4px 4px 0;
                }
                
                .citation-text {
                    color: #495057;
                }
                
                .summary-card {
                    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                    border-radius: 8px;
                    padding: 20px;
                    margin: 15px 0;
                    border-left: 4px solid #1f77b4;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .improvement-card {
                    background: white;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    height: 100%;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .improvement-card h5 {
                    color: #1f77b4;
                    margin-bottom: 10px;
                    border-bottom: 2px solid #f0f0f0;
                    padding-bottom: 5px;
                }
                
                .improvement-list {
                    margin-top: 10px;
                }
                
                .improvement-item {
                    padding: 5px 0;
                    border-bottom: 1px solid #f0f0f0;
                }
                
                .improvement-item:last-child {
                    border-bottom: none;
                }
                </style>
            """, unsafe_allow_html=True)

        with tabs[3]:
            st.header("Transcript with Timestamps")
            transcript = evaluation.get("transcript", "")
            
            # Split transcript into sentences and add timestamps
            sentences = re.split(r'(?<=[.!?])\s+', transcript)
            for i, sentence in enumerate(sentences):
                # Calculate approximate timestamp based on words and average speaking rate
                words_before = len(' '.join(sentences[:i]).split())
                timestamp = words_before / 150  # Assuming 150 words per minute
                minutes = int(timestamp)
                seconds = int((timestamp - minutes) * 60)
                
                st.markdown(f"**[{minutes:02d}:{seconds:02d}]** {sentence}")

            # Comment out original transcript display
            # st.text(evaluation.get("transcript", "Transcript not available"))

    except Exception as e:
        logger.error(f"Error displaying evaluation: {e}")
        st.error(f"Error displaying results: {str(e)}")
        st.error("Please check the evaluation data structure and try again.")

    # Add these styles to the existing CSS in the main function
    st.markdown("""
        <style>
        /* ... existing styles ... */
        
        .citation-box {
            background-color: #f8f9fa;
            border-left: 3px solid #6c757d;
            padding: 10px;
            margin: 5px 0;
            border-radius: 0 4px 4px 0;
        }
        
        .recommendation-card {
            background-color: #ffffff;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .recommendation-card h4 {
            color: #1f77b4;
            margin: 0 0 10px 0;
        }
        
        .rigor-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 20px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .score-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .green-score {
            background-color: #28a745;
            color: white;
        }
        
        .orange-score {
            background-color: #fd7e14;
            color: white;
        }
        
        .metric-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .profile-guide {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #1f77b4;
        }
        
        .profile-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .profile-card.recommended {
            border-left: 4px solid #28a745;
        }
        
        .profile-header {
            margin-bottom: 15px;
        }
        
        .profile-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            margin-top: 5px;
            background-color: #f8f9fa;
        }
        
        .profile-content ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .recommendation-status {
            margin-top: 15px;
            padding: 10px;
            border-radius: 4px;
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        .recommendation-status small {
            display: block;
            margin-top: 5px;
            font-weight: normal;
            color: #666;
        }
        
        .recommendation-status.recommended {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
        }
        
        .recommendation-status:not(.recommended) {
            background-color: #fff3cd;
            border-color: #ffeeba;
            color: #856404;
        }
        
        .profile-card.recommended {
            border-left: 4px solid #28a745;
            box-shadow: 0 2px 8px rgba(40, 167, 69, 0.1);
        }
        
        .profile-card:not(.recommended) {
            border-left: 4px solid #ffc107;
            opacity: 0.8;
        }
        
        .profile-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .progress-metric {
            background: linear-gradient(135deg, #f6f8fa 0%, #ffffff 100%);
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            margin: 5px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        
        .progress-metric:hover {
            transform: translateX(5px);
        }
        
        .progress-metric b {
            color: #1f77b4;
        }
        
        /* Enhanced status messages */
        .status-message {
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
            animation: fadeIn 0.5s ease;
        }
        
        .status-processing {
            background: linear-gradient(135deg, #f0f7ff 0%, #e5f0ff 100%);
            border-left: 4px solid #1f77b4;
        }
        
        .status-complete {
            background: linear-gradient(135deg, #f0fff0 0%, #e5ffe5 100%);
            border-left: 4px solid #28a745;
        }
        
        .status-error {
            background: linear-gradient(135deg, #fff0f0 0%, #ffe5e5 100%);
            border-left: 4px solid #dc3545;
        }
        
        /* Progress bar enhancement */
        .stProgress > div > div {
            background-image: linear-gradient(
                to right,
                rgba(31, 119, 180, 0.8),
                rgba(31, 119, 180, 1)
            );
            transition: width 0.3s ease;
        }
        
        /* Batch indicator animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .batch-indicator {
            display: inline-block;
            padding: 4px 8px;
            background: #1f77b4;
            color: white;
            border-radius: 4px;
            animation: pulse 1s infinite;
        }
        
        .metric-box {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            padding: 10px;
            border-radius: 8px;
            margin: 5px;
            border-left: 4px solid #1f77b4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        
        .metric-box:hover {
            transform: translateX(5px);
        }
        
        .metric-box.batch {
            border-left-color: #28a745;
        }
        
        .metric-box.time {
            border-left-color: #dc3545;
        }
        
        .metric-box.progress {
            border-left-color: #ffc107;
        }
        
        .metric-box.segment {
            border-left-color: #17a2b8;
        }
        
        .metric-box b {
            color: #1f77b4;
        }
        
        <style>
        .metric-explanation-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #17a2b8;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .metric-explanation-card h4 {
            color: #17a2b8;
            margin-bottom: 10px;
        }
        
        .metric-explanation-card ul {
            list-style-type: none;
            padding-left: 0;
        }
        
        .metric-explanation-card li {
            margin-bottom: 12px;
            padding-left: 15px;
            border-left: 2px solid #e9ecef;
        }
        
        .metric-explanation-card li:hover {
            border-left: 2px solid #17a2b8;
        }
        </style>
        
        <style>
        /* ... existing styles ... */
        
        .suggestions-box {
            background-color: #f8f9fa;
            padding: 10px 15px;
            margin-top: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }
        
        .suggestions-box h4 {
            color: #856404;
            margin: 0;
            padding: 5px 0;
        }
        
        .suggestion-item {
            padding: 5px 15px;
            color: #666;
            border-left: 2px solid #ffc107;
            margin: 5px 0;
            background-color: #fff;
            border-radius: 0 4px 4px 0;
        }
        
        .suggestion-item:hover {
            background-color: #fff9e6;
            transform: translateX(5px);
            transition: all 0.2s ease;
        }
        </style>
    """, unsafe_allow_html=True)

def check_dependencies() -> List[str]:
    """Check if required dependencies are installed"""
    missing = []
    
    if not shutil.which('ffmpeg'):
        missing.append("FFmpeg")
    
    return missing

def generate_pdf_report(evaluation_data: Dict[str, Any]) -> bytes:
    """Generate a formatted PDF report from evaluation data"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from io import BytesIO
        
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Mentor Demo Evaluation Report", title_style))
        story.append(Spacer(1, 20))
        
        # Communication Metrics Section
        story.append(Paragraph("Communication Metrics", styles['Heading2']))
        comm_metrics = evaluation_data.get("communication", {})
        
        # Create tables for each metric category
        for category in ["speed", "fluency", "flow", "intonation", "energy"]:
            if category in comm_metrics:
                metrics = comm_metrics[category]
                story.append(Paragraph(category.title(), styles['Heading3']))
                
                data = [[k.replace('_', ' ').title(), str(v)] for k, v in metrics.items()]
                t = Table(data, colWidths=[200, 200])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(t)
                story.append(Spacer(1, 20))
        
        # Teaching Analysis Section
        story.append(Paragraph("Teaching Analysis", styles['Heading2']))
        teaching_data = evaluation_data.get("teaching", {})
        
        for assessment_type in ["Concept Assessment", "Code Assessment"]:
            if assessment_type in teaching_data:
                story.append(Paragraph(assessment_type, styles['Heading3']))
                categories = teaching_data[assessment_type]
                
                for category, details in categories.items():
                    score = details.get("Score", 0)
                    citations = details.get("Citations", [])
                    
                    data = [
                        [category, "Score: " + ("Pass" if score == 1 else "Needs Improvement")],
                        ["Citations:", ""]
                    ] + [["-", citation] for citation in citations]
                    
                    t = Table(data, colWidths=[200, 300])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 20))
        
        # Recommendations Section
        story.append(Paragraph("Recommendations", styles['Heading2']))
        recommendations = evaluation_data.get("recommendations", {})
        
        if "summary" in recommendations:
            story.append(Paragraph("Overall Summary:", styles['Heading3']))
            story.append(Paragraph(recommendations["summary"], styles['Normal']))
            story.append(Spacer(1, 20))
        
        if "improvements" in recommendations:
            story.append(Paragraph("Areas for Improvement:", styles['Heading3']))
            improvements = recommendations["improvements"]
            for improvement in improvements:
                # Handle both string and dictionary improvement formats
                if isinstance(improvement, dict):
                    message = improvement.get("message", "")
                    category = improvement.get("category", "")
                    story.append(Paragraph(f"• [{category}] {message}", styles['Normal']))
                else:
                    story.append(Paragraph(f"• {improvement}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        raise RuntimeError(f"Failed to generate PDF report: {str(e)}")

def main():
    try:
        # Set page config must be the first Streamlit command
        st.set_page_config(page_title="🎓 Mentor Demo Review System", layout="wide")
        
        # Initialize session state for tracking progress
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
        
        # Add custom CSS for animations and styling
        st.markdown("""
            <style>
                /* Shimmer animation keyframes */
                @keyframes shimmer {
                    0% {
                        background-position: -1000px 0;
                    }
                    100% {
                        background-position: 1000px 0;
                    }
                }
                
                .title-shimmer {
                    text-align: center;
                    color: #1f77b4;
                    position: relative;
                    overflow: hidden;
                    background: linear-gradient(
                        90deg,
                        rgba(255, 255, 255, 0) 0%,
                        rgba(255, 255, 255, 0.8) 50%,
                        rgba(255, 255, 255, 0) 100%
                    );
                    background-size: 1000px 100%;
                    animation: shimmer 3s infinite linear;
                }
                
                /* Existing animations */
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                
                @keyframes slideIn {
                    from { transform: translateX(-100%); }
                    to { transform: translateX(0); }
                }
                
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }
                
                .fade-in {
                    animation: fadeIn 1s ease-in;
                }
                
                .slide-in {
                    animation: slideIn 0.5s ease-out;
                }
                
                .pulse {
                    animation: pulse 2s infinite;
                }
                
                .metric-card {
                    background-color: #f0f2f6;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    transition: transform 0.3s ease;
                }
                
                .metric-card:hover {
                    transform: translateY(-5px);
                }
                
                .stButton>button {
                    transition: all 0.3s ease;
                }
                
                .stButton>button:hover {
                    transform: scale(1.05);
                }
                
                .category-header {
                    background: linear-gradient(90deg, #1f77b4, #2c3e50);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                
                .score-badge {
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-weight: bold;
                }
                
                .score-pass {
                    background-color: #28a745;
                    color: white;
                }
                
                .score-fail {
                    background-color: #dc3545;
                    color: white;
                }
                
                .metric-box {
                    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                    padding: 10px;
                    border-radius: 8px;
                    margin: 5px;
                    border-left: 4px solid #1f77b4;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: transform 0.2s ease;
                }
                
                .metric-box:hover {
                    transform: translateX(5px);
                }
                
                .metric-box.batch {
                    border-left-color: #28a745;
                }
                
                .metric-box.time {
                    border-left-color: #dc3545;
                }
                
                .metric-box.progress {
                    border-left-color: #ffc107;
                }
                
                .metric-box.segment {
                    border-left-color: #17a2b8;
                }
                
                .metric-box b {
                    color: #1f77b4;
                }
            </style>
            
            <div class="fade-in">
                <h1 class="title-shimmer">
                    🎓 Mentor Demo Review System
                </h1>
            </div>
        """, unsafe_allow_html=True)

        # Sidebar with instructions and status
        with st.sidebar:
            st.markdown("""
                <div class="slide-in">
                    <h2>Instructions</h2>
                    <ol>
                        <li>Upload your teaching video</li>
                        <li>Wait for the analysis</li>
                        <li>Review the detailed feedback</li>
                        <li>Download the report</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
            
            # Add file format information separately
            st.markdown("**Supported formats:** MP4, AVI, MOV")
            st.markdown("**Maximum file size:** 1GB")
            
            # Create a placeholder for status updates in the sidebar
            status_placeholder = st.empty()
            status_placeholder.info("Upload a video to begin analysis")

        # Check dependencies with progress
        with st.status("Checking system requirements...") as status:
            progress_bar = st.progress(0)
            
            status.update(label="Checking FFmpeg installation...")
            progress_bar.progress(0.3)
            missing_deps = check_dependencies()
            
            progress_bar.progress(0.6)
            if missing_deps:
                status.update(label="Missing dependencies detected!", state="error")
                st.error(f"Missing required dependencies: {', '.join(missing_deps)}")
                st.markdown("""
                Please install the missing dependencies:
                ```bash
                sudo apt-get update
                sudo apt-get install ffmpeg
                ```
                """)
                return
            
            progress_bar.progress(1.0)
            status.update(label="System requirements satisfied!", state="complete")

        # Add input selection with improved styling
        st.markdown("""
            <style>
            .input-selection {
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #1f77b4;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .upload-section {
                background: #ffffff;
                padding: 20px;
                border-radius: 8px;
                margin-top: 15px;
                border: 1px solid #e0e0e0;
            }
            
            .upload-header {
                color: #1f77b4;
                font-size: 1.2em;
                margin-bottom: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Input type selection with better UI
        st.markdown('<div class="input-selection">', unsafe_allow_html=True)
        st.markdown("### 📤 Select Upload Method")
        input_type = st.radio(
            "Choose how you want to provide your teaching content:",
            options=[
                "Video Only (Auto-transcription)",
                "Video + Manual Transcript"
            ],
            help="Select whether you want to upload just the video (we'll transcribe it) or provide your own transcript"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Video upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<p class="upload-header">📹 Upload Teaching Video</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload your teaching video (MP4, AVI, or MOV format, max 1GB)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Transcript upload section (conditional)
        uploaded_transcript = None
        if input_type == "Video + Manual Transcript":
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown('<p class="upload-header">📝 Upload Transcript</p>', unsafe_allow_html=True)
            uploaded_transcript = st.file_uploader(
                "Select transcript file",
                type=['txt'],
                help="Upload your transcript (TXT format)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Process video when uploaded
        if uploaded_file:
            if input_type == "Video + Manual Transcript" and not uploaded_transcript:
                st.warning("Please upload both video and transcript files to continue.")
                return
                
            # Only process if not already completed
            if not st.session_state.processing_complete:
                status_placeholder.info("Video uploaded, beginning processing...")
                
                st.markdown("""
                    <div class="pulse" style="text-align: center;">
                        <h3>Processing your video...</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create temp directory for processing
                temp_dir = tempfile.mkdtemp()
                video_path = os.path.join(temp_dir, uploaded_file.name)
                
                try:
                    # Save uploaded file with progress
                    with st.status("Saving uploaded file...") as status:
                        # Update sidebar status
                        status_placeholder.info("Saving uploaded file...")
                        progress_bar = st.progress(0)
                        
                        # Save in chunks to show progress
                        chunk_size = 1024 * 1024  # 1MB chunks
                        file_size = len(uploaded_file.getbuffer())
                        chunks = file_size // chunk_size + 1
                        
                        with open(video_path, 'wb') as f:
                            for i in range(chunks):
                                start = i * chunk_size
                                end = min(start + chunk_size, file_size)
                                f.write(uploaded_file.getbuffer()[start:end])
                                progress = (i + 1) / chunks
                                status.update(label=f"Saving file: {progress:.1%}")
                                progress_bar.progress(progress)
                        
                        status.update(label="File saved successfully!", state="complete")
                    
                    # Validate file size
                    file_size = os.path.getsize(video_path) / (1024 * 1024 * 1024)
                    if file_size > 1:
                        st.error("File size exceeds 1GB limit. Please upload a smaller file.")
                        return
                    
                    # Process video
                    status_placeholder.info("Processing video and generating analysis...")
                    
                    process_container = st.container()
                    with process_container:
                        st.markdown("""
                            <div class="processing-status">
                                <h3>🎥 Processing Video</h3>
                                <div class="status-details"></div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        evaluator = MentorEvaluator()
                        st.session_state.evaluation_results = evaluator.evaluate_video(
                            video_path,
                            uploaded_transcript if input_type == "Video + Manual Transcript" else None
                        )
                        st.session_state.processing_complete = True
                        
                except Exception as e:
                    status_placeholder.error(f"Error during processing: {str(e)}")
                    st.error(f"Error during evaluation: {str(e)}")
                    
                finally:
                    # Clean up temp files
                    if 'temp_dir' in locals():
                        shutil.rmtree(temp_dir)
            
            # Display results if processing is complete
            if st.session_state.processing_complete and st.session_state.evaluation_results:
                status_placeholder.success("Analysis complete! Review results below.")
                st.success("Analysis complete!")
                display_evaluation(st.session_state.evaluation_results)
                
                # Add download options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.download_button(
                        "📥 Download JSON Report",
                        json.dumps(st.session_state.evaluation_results, indent=2),
                        "evaluation_report.json",
                        "application/json",
                        help="Download the raw evaluation data in JSON format"
                    ):
                        st.success("JSON report downloaded successfully!")
                
                with col2:
                    if st.download_button(
                        "📄 Download Full Report (PDF)",
                        generate_pdf_report(st.session_state.evaluation_results),
                        "evaluation_report.pdf",
                        "application/pdf",
                        help="Download a formatted PDF report with detailed analysis"
                    ):
                        st.success("PDF report downloaded successfully!")

    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
