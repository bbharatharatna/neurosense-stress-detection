# ============================================================
# NEUROSENSE - AI-BASED MULTIMODAL STRESS DETECTION SYSTEM
# Full-Featured Version | 5 Modalities | 67+ Features
#
# Team 17 | IT-D | 4th B.Tech II-SEM | Anurag University
# Members: A Raviteja, B Bharatha Ratna, J Nagesh
# Mentor : Mr. Mahesh Kumar Swamy
# ============================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import tempfile
import os
import wave
import json
import csv
import io
from scipy.signal import find_peaks
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from st_audiorec import st_audiorec
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NeuroSense - AI Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
def load_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); }
    h1 {
        color: #ffd700; text-align: center; padding: 20px;
        background: rgba(255,215,0,0.1); border-radius: 10px;
        margin-bottom: 30px; border: 2px solid #ffd700;
    }
    h2, h3 { color: #ffd700; }
    .success-box {
        background: #1a2e1a; border: 2px solid #ffd700;
        border-radius: 10px; padding: 15px; color: #90ee90; margin: 10px 0;
    }
    .warning-box {
        background: #3d3d2d; border: 2px solid #ffd700;
        border-radius: 10px; padding: 15px; color: #ffd700; margin: 10px 0;
    }
    .danger-box {
        background: #3d2d2d; border: 2px solid #ffd700;
        border-radius: 10px; padding: 15px; color: #ff6b6b; margin: 10px 0;
    }
    .eeg-box {
        background: #1a1a2e; border: 2px solid #7b68ee;
        border-radius: 10px; padding: 15px; color: #a0a0ff; margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #ffd700, #ffed4e);
        color: #1a1a1a; border: none; padding: 15px;
        border-radius: 8px; font-size: 16px; font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,215,0,0.4);
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# EEG STRESS ANALYZER
# ============================================================
class EEGStressAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = []

    def load_and_train(self, csv_path: str) -> dict:
        try:
            df = pd.read_csv(csv_path)
            label_col = None
            for c in ['label', 'Label', 'stress', 'Stress', 'class', 'Class', 'emotion', 'Emotion']:
                if c in df.columns:
                    label_col = c
                    break
            if label_col is None:
                label_col = df.columns[-1]

            feature_cols = [c for c in df.columns if c != label_col]
            X_raw = df[feature_cols].select_dtypes(include=[np.number])
            y_raw = df[label_col].copy()

            if y_raw.nunique() > 2:
                relaxed_lbl = y_raw.mode()[0]
                y_raw = (y_raw != relaxed_lbl).astype(int)
            else:
                y_raw = y_raw.astype(int)

            mask = X_raw.notna().all(axis=1) & y_raw.notna()
            X_raw = X_raw[mask].reset_index(drop=True)
            y_raw = y_raw[mask].reset_index(drop=True)

            X_feat, self.feature_names = self._extract_features(X_raw)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_feat, y_raw, test_size=0.2, random_state=42, stratify=y_raw
            )

            self.scaler = StandardScaler()
            X_tr_s = self.scaler.fit_transform(X_tr)
            X_te_s = self.scaler.transform(X_te)

            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            self.model.fit(X_tr_s, y_tr)
            self.is_trained = True

            return {
                'success': True,
                'train_accuracy': self.model.score(X_tr_s, y_tr),
                'test_accuracy': self.model.score(X_te_s, y_te),
                'samples': len(X_feat),
                'features': len(self.feature_names),
                'label_column': label_col,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def predict_eeg_stress(self, eeg_row: np.ndarray):
        if not self.is_trained:
            return 0.5, "EEG model not trained. Upload emotions.csv first.", None, 0.0
        try:
            row_df = pd.DataFrame([eeg_row])
            X_feat, _ = self._extract_features(row_df)
            X_scaled = self.scaler.transform(X_feat)
            proba = self.model.predict_proba(X_scaled)[0]
            stress_p = float(proba[1])
            conf = float(np.max(proba))
            feat_dict = dict(zip(self.feature_names, X_feat[0]))
            status = (
                f"Alpha/Beta: {feat_dict.get('eeg_alpha_beta_ratio', 0):.3f} | "
                f"Spec.Entropy: {feat_dict.get('eeg_spectral_entropy', 0):.2f} | "
                f"Beta Power: {feat_dict.get('eeg_beta', 0):.2f}"
            )
            return stress_p, status, feat_dict, conf
        except Exception as e:
            return 0.5, f"Prediction error: {str(e)[:60]}", None, 0.0

    def get_random_sample_prediction(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path)
            label_col = None
            for c in ['label', 'Label', 'stress', 'Stress', 'class', 'Class', 'emotion', 'Emotion']:
                if c in df.columns:
                    label_col = c
                    break
            if label_col is None:
                label_col = df.columns[-1]

            feat_cols = [c for c in df.columns if c != label_col]
            X_raw = df[feat_cols].select_dtypes(include=[np.number])
            idx = np.random.randint(0, len(X_raw))
            sample = X_raw.iloc[idx].values
            actual = df[label_col].iloc[idx]
            s, st_msg, fd, c = self.predict_eeg_stress(sample)
            return s, st_msg, fd, c, actual
        except Exception as e:
            return 0.5, f"Sample error: {str(e)[:60]}", None, 0.0, None

    def _extract_features(self, X_df: pd.DataFrame):
        results, names, built = [], [], False
        FS = 128

        for _, row in X_df.iterrows():
            sig = row.values.astype(float)
            f = []

            mu = np.mean(sig)
            sd = np.std(sig) + 1e-10
            f += [float(mu), float(sd), float(np.var(sig)), float(np.max(sig) - np.min(sig))]
            f.append(float(np.mean(((sig - mu) / sd) ** 3)))
            f.append(float(np.mean(((sig - mu) / sd) ** 4) - 3))

            freqs = np.fft.rfftfreq(len(sig), d=1.0 / FS)
            psd = np.abs(np.fft.rfft(sig)) ** 2

            def band_power(lo, hi, freqs=freqs, psd=psd):
                idx = np.where((freqs >= lo) & (freqs < hi))[0]
                return float(np.mean(psd[idx])) if len(idx) else 0.0

            delta = band_power(0.5, 4)
            theta = band_power(4, 8)
            alpha = band_power(8, 13)
            beta = band_power(13, 30)
            gamma = band_power(30, 45)
            f += [delta, theta, alpha, beta, gamma]

            f.append(float(alpha / max(beta, 1e-10)))
            f.append(float(theta / max(alpha, 1e-10)))

            psd_n = psd / (psd.sum() + 1e-10)
            psd_n = psd_n[psd_n > 0]
            f.append(float(-np.sum(psd_n * np.log2(psd_n))))
            f.append(float(psd.sum()))

            d1 = np.diff(sig)
            d2 = np.diff(d1)
            act = float(np.var(sig))
            mob = float(np.std(d1) / max(np.std(sig), 1e-10))
            cmp = float((np.std(d2) / max(np.std(d1), 1e-10)) / max(mob, 1e-10))
            f += [act, mob, cmp]

            results.append(f)

            if not built:
                names = [
                    'eeg_mean', 'eeg_std', 'eeg_variance', 'eeg_range',
                    'eeg_skewness', 'eeg_kurtosis',
                    'eeg_delta', 'eeg_theta', 'eeg_alpha', 'eeg_beta', 'eeg_gamma',
                    'eeg_alpha_beta_ratio', 'eeg_theta_alpha_ratio',
                    'eeg_spectral_entropy', 'eeg_total_power',
                    'eeg_hjorth_activity', 'eeg_hjorth_mobility', 'eeg_hjorth_complexity'
                ]
                built = True

        return np.array(results), names


# ============================================================
# ENHANCED STRESS DETECTION
# ============================================================
class EnhancedStressDetectionSystem:

    def __init__(self):
        self.stress_keywords = {
            'panic': 1.0, 'overwhelmed': 1.0, 'terrified': 0.9,
            'anxious': 0.8, 'anxiety': 0.8, 'stressed': 0.7,
            'worried': 0.7, 'worry': 0.7, 'nervous': 0.7,
            'frustrated': 0.7, 'angry': 0.7, 'exhausted': 0.8,
            'tired': 0.6, 'fatigue': 0.7, 'pressure': 0.6,
            'deadline': 0.6, 'tense': 0.7, 'depressed': 0.9,
            'sad': 0.6, 'hopeless': 0.9, 'afraid': 0.7,
            'insomnia': 0.8, 'difficult': 0.5, 'concerned': 0.5,
            'overwhelm': 0.9, 'burnout': 0.9, 'distressed': 0.8,
            'miserable': 0.8, 'helpless': 0.9, 'restless': 0.6,
        }
        self.positive_keywords = {
            'excellent': 0.9, 'amazing': 0.9, 'wonderful': 0.9,
            'happy': 0.8, 'happiness': 0.8, 'great': 0.7,
            'good': 0.6, 'relaxed': 0.9, 'calm': 0.9,
            'peaceful': 0.9, 'comfortable': 0.7, 'confident': 0.8,
            'positive': 0.7, 'energetic': 0.7, 'motivated': 0.8,
            'excited': 0.7, 'joyful': 0.8, 'content': 0.8,
            'grateful': 0.8, 'productive': 0.7, 'focused': 0.7,
        }
        self.negation_words = [
            'not', 'no', 'never', "don't", "didn't",
            "won't", "cannot", "can't", "isn't", "wasn't"
        ]

    def predict_facial_emotion(self, image):
        try:
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml')

            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            if len(faces) == 0:
                return 0.5, "No face detected", None, 0.0

            (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
            face_roi = gray[y:y + h, x:x + w]
            features = {}

            fl = face_roi[:, :w // 2]
            fr = cv2.flip(face_roi[:, w // 2:], 1)
            mw = min(fl.shape[1], fr.shape[1])
            sym_diff = float(np.mean(np.abs(fl[:, :mw].astype(float) - fr[:, :mw].astype(float))))
            features['facial_symmetry'] = sym_diff

            brightness = float(np.mean(face_roi))
            eye_brightness = float(np.mean(face_roi[:int(h * 0.4), :]))
            features['brightness'] = brightness
            features['eye_brightness'] = eye_brightness

            lap = cv2.Laplacian(face_roi, cv2.CV_64F)
            tex_v = float(np.var(lap))
            features['texture_variance'] = tex_v

            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)
            num_eyes = len(eyes)
            features['num_eyes'] = num_eyes

            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = float(np.sum(edges > 0) / edges.size)
            fh_edges = cv2.Canny(face_roi[:int(h * 0.3), :], 50, 150)
            fh_density = float(np.sum(fh_edges > 0) / fh_edges.size)
            features['edge_density'] = edge_density
            features['forehead_edges'] = fh_density

            face_std = float(np.std(face_roi))
            features['contrast'] = face_std
            features['face_size'] = float(w * h / (gray.shape[0] * gray.shape[1]))

            fc = opencv_image[y:y + h, x:x + w]
            b_ch, g_ch, r_ch = cv2.split(fc)
            features['color_r'] = float(np.mean(r_ch))
            features['color_g'] = float(np.mean(g_ch))
            features['color_b'] = float(np.mean(b_ch))
            features['surface_roughness'] = float(tex_v / max(brightness, 1))

            sym_s = min(sym_diff / 50, 1.0)
            bri_s = 1.0 - (brightness / 255)
            tex_s = min(tex_v / 1000, 1.0)
            edg_s = min(edge_density * 10, 1.0)
            fh_s = min(fh_density * 15, 1.0)
            con_s = min(face_std / 100, 1.0)
            eye_f = 0.0 if num_eyes == 2 else 0.3

            facial_stress = float(
                sym_s * 0.20 + bri_s * 0.15 + tex_s * 0.20 +
                edg_s * 0.15 + fh_s * 0.15 + con_s * 0.10 + eye_f * 0.05
            )
            facial_stress = max(0.0, min(1.0, facial_stress))

            conf = float(
                (1.0 if len(faces) == 1 else 0.7) *
                min(w / 100, 1.0) *
                (1.0 if num_eyes == 2 else 0.5)
            )
            status = f"Face: {w}x{h}px | Eyes: {num_eyes} | Symmetry: {sym_diff:.1f}"
            return facial_stress, status, features, conf

        except Exception as e:
            return 0.5, f"Error: {str(e)}", None, 0.0

    def predict_voice_emotion(self, audio_bytes):
        if audio_bytes is None:
            return 0.5, "No audio provided", None, 0.0
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with wave.open(tmp_path, 'rb') as wf:
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                fs = wf.getframerate()
                nf = wf.getnframes()
                frames = wf.readframes(nf)

            audio = np.frombuffer(frames, dtype=np.int16 if sw == 2 else np.uint8)
            if n_ch == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)
            audio = audio.astype(float)
            duration = float(nf / fs)

            if duration < 0.5:
                os.unlink(tmp_path)
                return 0.5, "Audio too short (need 0.5s+)", None, 0.0

            features = {'duration': duration}
            frame_size = int(0.025 * fs)
            hop_size = int(0.010 * fs)
            pitches, energies, zcrs = [], [], []

            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                rms = float(np.sqrt(np.mean(frame ** 2)))
                energies.append(rms)
                zcrs.append(float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))))

                if rms > 100:
                    ac = np.correlate(frame, frame, mode='full')
                    ac = ac[len(ac) // 2:]
                    if ac[0] > 0:
                        ac = ac / ac[0]
                        lo = int(fs / 400)
                        hi = int(fs / 50)
                        if len(ac) > hi:
                            pks, _ = find_peaks(ac[lo:hi])
                            if len(pks):
                                p = float(fs / (pks[0] + lo))
                                if 50 <= p <= 400:
                                    pitches.append(p)

            if pitches:
                pm = float(np.mean(pitches))
                ps_val = float(np.std(pitches))
                pv = float(ps_val / max(pm, 1))
            else:
                pm, ps_val, pv = 150.0, 0.0, 0.0

            features.update({'pitch_mean': pm, 'pitch_std': ps_val,
                              'pitch_variation': pv, 'pitch_segments': len(pitches)})

            em = float(np.mean(energies))
            es = float(np.std(energies))
            ev = float(es / max(em, 1))
            features.update({'energy_mean': em, 'energy_std': es, 'energy_variation': ev})

            zcr_mean = float(np.mean(zcrs))
            features['zcr_mean'] = zcr_mean

            sil_thr = em * 0.3
            pauses = 0
            in_p = False
            for e_val in energies:
                if e_val < sil_thr:
                    if not in_p:
                        pauses += 1
                        in_p = True
                else:
                    in_p = False
            features['pauses'] = pauses

            amp_max = float(np.max(np.abs(audio)))
            amp_mean = float(np.mean(np.abs(audio)))
            features.update({'amplitude_max': amp_max, 'amplitude_mean': amp_mean,
                              'dynamic_range': float(amp_max / max(amp_mean, 1))})

            pv_n = min(pv, 1.0)
            ev_n = min(ev, 1.0)
            zcr_n = min(zcr_mean * 2, 1.0)
            pa_n = min(pauses / 20, 1.0)

            voice_stress = float(pv_n * 0.30 + ev_n * 0.30 + zcr_n * 0.20 + pa_n * 0.20)
            voice_stress = max(0.0, min(1.0, voice_stress))

            conf = float(
                min(duration / 2, 1.0) *
                min(len(pitches) / 3, 1.0) *
                min(amp_mean / 500, 1.0)
            )

            os.unlink(tmp_path)
            status = f"Duration: {duration:.1f}s | Pitch: {pm:.0f}Hz | Pauses: {pauses}"
            return voice_stress, status, features, conf

        except Exception as e:
            return 0.5, f"Error: {str(e)[:50]}", None, 0.0

    def predict_text_sentiment(self, text):
        if not text or not text.strip():
            return 0.5, None, 0.0

        tc = text.strip()
        wds = tc.split()
        wc = len(wds)
        features = {}
        features['char_count'] = len(tc)
        features['word_count'] = wc

        if wc == 0:
            return 0.5, features, 0.0

        sc = max(tc.count('.') + tc.count('!') + tc.count('?'), 1)
        features['sentence_count'] = sc
        features['avg_sentence_length'] = float(wc / sc)

        wl = tc.lower().split()
        ss = 0.0
        ps = 0.0
        sm = 0.0
        pm_kw = 0.0

        for i, w in enumerate(wl):
            start = max(0, i - 3)
            prev_words = wl[start:i]
            neg = any(pw in self.negation_words for pw in prev_words)

            if w in self.stress_keywords:
                wt = self.stress_keywords[w]
                if neg:
                    ps += wt
                else:
                    ss += wt
                    sm += 1
            if w in self.positive_keywords:
                wt = self.positive_keywords[w]
                if neg:
                    ss += wt
                else:
                    ps += wt
                    pm_kw += 1

        features.update({'stress_keywords': sm, 'positive_keywords': pm_kw,
                          'stress_raw': ss, 'positive_raw': ps})

        exc = tc.count('!')
        qst = tc.count('?')
        ell = tc.count('...')
        features.update({'exclamations': exc, 'questions': qst, 'ellipsis': ell})

        cr = float(sum(1 for c in tc if c.isupper()) / max(len(tc), 1))
        ur = float(len(set(wl)) / max(wc, 1))
        features.update({'caps_ratio': cr, 'unique_word_ratio': ur})

        tot = ss + ps
        ksr = float(ss / tot) if tot > 0 else 0.5
        exc_s = min(exc / 5, 0.3)
        cap_s = min(cr, 0.3)
        rep_s = 1.0 - ur
        cmp_s = 1.0 - min(float(wc / sc) / 15, 1.0)

        ts = float(ksr * 0.50 + exc_s * 0.15 + cap_s * 0.15 + rep_s * 0.10 + cmp_s * 0.10)
        ts = max(0.0, min(1.0, ts))

        conf = float(
            min(wc / 10, 1.0) *
            min((sm + pm_kw) / 3, 1.0) *
            min(sc / 2, 1.0)
        )
        return ts, features, conf

    def predict_survey_stress(self, sleep, workload, mood):
        sl_s = ((6 - sleep) / 5.0) ** 1.2
        wl_s = (workload / 5.0) ** 1.1
        md_s = ((6 - mood) / 5.0) ** 1.2
        s = sl_s * 0.35 + wl_s * 0.35 + md_s * 0.30
        return max(0.0, min(1.0, float(s)))

    def calculate_overall_stress(self,
                                  facial, voice, text, survey,
                                  fc, vc, tc, sc=1.0,
                                  eeg=0.5, ec=0.0):
        num = facial * fc + voice * vc + text * tc + survey * sc + eeg * ec
        den = fc + vc + tc + sc + ec
        if den == 0:
            return 0.5, 0.0
        os_ = float(num / den)
        oc = float(den / 5.0)
        return max(0.0, min(1.0, os_)), max(0.0, min(1.0, oc))

    def generate_recommendations(self, level, facial, voice, text):
        if level == 'Low':
            return [
                "Your stress levels are healthy! Maintain your current routine.",
                "Practice gratitude - write down 3 things you are thankful for today.",
                "Continue regular physical activity to stay energised.",
                "Consider learning a new skill for personal growth.",
                "Stay connected with friends and family.",
            ]
        elif level == 'Moderate':
            recs = []
            if facial >= 0.5:
                recs += ["HIGH FACIAL TENSION: Practice progressive facial relaxation.",
                         "Try facial muscle release exercises in front of a mirror."]
            if voice >= 0.5:
                recs += ["VOICE STRESS: Practice 4-7-8 breathing (Inhale 4, Hold 7, Exhale 8).",
                         "Listen to calming music or nature sounds."]
            if text >= 0.5:
                recs += ["LINGUISTIC STRESS: Journal your thoughts to process emotions.",
                         "Practice positive affirmations daily."]
            recs += [
                "Take regular breaks every 50 minutes during work.",
                "Go for a 15-minute walk outdoors.",
                "Stay hydrated - drink 8 glasses of water daily.",
                "Limit screen time 1 hour before bedtime.",
                "10 minutes of meditation or deep breathing daily.",
            ]
            return recs[:7]
        else:
            return [
                "HIGH STRESS ALERT - Take immediate action NOW.",
                "Reach out to a trusted friend, family member, or counsellor immediately.",
                "Strongly consider consulting a mental health professional today.",
                "Practice 4-7-8 breathing RIGHT NOW for 5 minutes.",
                "Step away from stressors - take an immediate break (15-30 min).",
                "Prioritise sleep tonight - aim for 7-9 hours.",
                "Avoid caffeine and alcohol for the next 24 hours.",
                "Use mental health apps: Calm, Headspace, or Insight Timer.",
                "Progressive muscle relaxation - tense/release each muscle group.",
                "Make a to-do list; tackle ONE task at a time.",
                "Do not isolate - connect with supportive people.",
                "",
                "CRISIS HELPLINES (India):",
                "  National Mental Health: 1800-599-0019",
                "  Vandrevala Foundation: 1860-2662-345",
                "  AASRA 24x7: 91-22-27546669",
            ]


# ============================================================
# PDF REPORT
# ============================================================
def generate_pdf_report(user_name, history):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font('Arial', 'B', 22)
    pdf.cell(0, 15, 'NeuroSense Stress Assessment Report', ln=True, align='C')
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Anurag University | AI-Based Multimodal Stress Detection', ln=True, align='C')
    pdf.ln(6)

    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 6, f"User: {user_name}", ln=True)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 6, f"Total Assessments: {len(history)}", ln=True)
    pdf.ln(6)

    if history:
        avg_s = sum(h['overall_stress'] for h in history) / len(history)
        avg_c = sum(h['overall_confidence'] for h in history) / len(history)
        low_c = sum(1 for h in history if h['level'] == 'Low')
        mod_c = sum(1 for h in history if h['level'] == 'Moderate')
        high_c = sum(1 for h in history if h['level'] == 'High')

        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 8, 'Summary Statistics:', ln=True)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 6, f"Average Stress Level : {avg_s:.1%}", ln=True)
        pdf.cell(0, 6, f"Average Confidence   : {avg_c:.1%}", ln=True)
        pdf.cell(0, 6, f"Low: {low_c} | Moderate: {mod_c} | High: {high_c}", ln=True)
        pdf.ln(5)

        avg_f = sum(h['breakdown']['facial'] for h in history) / len(history)
        avg_v = sum(h['breakdown']['voice'] for h in history) / len(history)
        avg_t = sum(h['breakdown']['text'] for h in history) / len(history)
        avg_sv = sum(h['breakdown']['survey'] for h in history) / len(history)
        avg_e = sum(h['breakdown'].get('eeg', 0.5) for h in history) / len(history)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Average by Modality:', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6,
                 f"Facial: {avg_f:.1%} | Voice: {avg_v:.1%} | "
                 f"Text: {avg_t:.1%} | Survey: {avg_sv:.1%} | EEG: {avg_e:.1%}",
                 ln=True)
        pdf.ln(5)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Recent Assessment History (Last 10):', ln=True)
        pdf.set_font('Arial', '', 10)
        for i, entry in enumerate(history[-10:], 1):
            line = (f"{i}. {entry['timestamp']} - "
                    f"{entry['overall_stress']:.1%} ({entry['level']}) - "
                    f"Conf: {entry['overall_confidence']:.1%}")
            pdf.cell(0, 6, line, ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(tmp.name)
    return tmp.name


# ============================================================
# SESSION STATE
# ============================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'detector' not in st.session_state:
    st.session_state.detector = EnhancedStressDetectionSystem()
if 'eeg_analyzer' not in st.session_state:
    st.session_state.eeg_analyzer = EEGStressAnalyzer()
if 'eeg_csv_path' not in st.session_state:
    st.session_state.eeg_csv_path = None
if 'eeg_score' not in st.session_state:
    st.session_state.eeg_score = 0.5
if 'eeg_conf' not in st.session_state:
    st.session_state.eeg_conf = 0.0
if 'eeg_features' not in st.session_state:
    st.session_state.eeg_features = None


# ============================================================
# MAIN APP
# ============================================================
def main():
    load_css()
    st.markdown("<h1>🧠 NeuroSense - AI Stress Detection</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Project Information")
        st.info("""
**NeuroSense - AI Stress Detection**
Anurag University

**Team 17 | IT-D | 4th B.Tech**
A Raviteja, B Bharatha Ratna, J Nagesh

**Mentor:** Mr. Mahesh Kumar Swamy

**Version:** Full-Featured (67+ Features)
**Modalities:** Facial, Voice, Text, Survey, EEG
        """)

        st.markdown("---")
        st.markdown("### Assessment History")
        if st.session_state.history:
            st.success(f"Total: {len(st.session_state.history)}")
            avg = sum(h['overall_stress'] for h in st.session_state.history) / len(st.session_state.history)
            st.metric("Average Stress", f"{avg:.1%}")
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.warning("No assessments yet")

        st.markdown("---")
        st.markdown("### EEG Model Status")
        if st.session_state.eeg_analyzer.is_trained:
            st.success("EEG Model Ready")
        else:
            st.warning("EEG not trained. Upload emotions.csv in EEG Setup tab.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Assessment", "Results History", "EEG Setup", "Features", "About"
    ])

    # ── TAB 1: ASSESSMENT ───────────────────────────────────
    with tab1:
        st.markdown("### Complete Multimodal Assessment (5 Modalities)")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Facial Expression Analysis")
            st.caption("15+ features: symmetry, texture, edges, brightness, eyes")
            img_file = st.camera_input("Capture your face")

            facial_score = 0.5
            facial_conf = 0.0
            facial_features = None

            if img_file:
                image = Image.open(img_file)
                facial_score, status, facial_features, facial_conf = \
                    st.session_state.detector.predict_facial_emotion(image)
                st.success(f"✅ {status}")
                st.metric("Facial Stress", f"{facial_score:.1%}",
                          help=f"Confidence: {facial_conf:.1%}")
                if facial_features:
                    with st.expander("Detailed Facial Features"):
                        for k, v in facial_features.items():
                            st.text(f"{k}: {v:.3f}")

            st.markdown("---")
            st.markdown("#### Voice Stress Analysis")
            st.caption("10+ features: pitch, energy, ZCR, pauses")

            voice_score = 0.5
            voice_conf = 0.0
            voice_features = None

            if RECORDER_AVAILABLE:
                st.info("Record 5-10 seconds of natural speech")
                audio_bytes = st_audiorec()
                if audio_bytes:
                    st.success("✅ Audio recorded!")
                    st.audio(audio_bytes, format='audio/wav')
                    voice_score, status, voice_features, voice_conf = \
                        st.session_state.detector.predict_voice_emotion(audio_bytes)
                    st.info(f"🎵 {status}")
                    st.metric("Voice Stress", f"{voice_score:.1%}",
                              help=f"Confidence: {voice_conf:.1%}")
                    if voice_features:
                        with st.expander("Detailed Voice Features"):
                            for k, v in voice_features.items():
                                st.text(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
            else:
                voice_score = 0.5
                voice_conf = 0.0
                st.warning("Install: pip install streamlit-audiorec")

        with col2:
            st.markdown("#### Text Sentiment Analysis")
            st.caption("13+ features: keywords, negation, complexity, punctuation")
            text_input = st.text_area(
                "Describe your current emotional state (50+ words recommended):",
                placeholder="I am feeling...",
                height=150
            )
            st.caption(f"Words: {len(text_input.split())} | Chars: {len(text_input)}")

            st.markdown("---")
            st.markdown("#### Lifestyle Survey")
            st.caption("Self-reported wellness factors")

            sleep = st.select_slider(
                "Sleep Quality (last night)",
                options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: ['Very Poor (<4h)', 'Poor (4-5h)', 'Fair (5-6h)',
                                        'Good (6-7h)', 'Excellent (7-9h)'][x - 1]
            )
            workload = st.select_slider(
                "Current Workload / Pressure",
                options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: ['Very Light', 'Light', 'Moderate', 'Heavy', 'Overwhelming'][x - 1]
            )
            mood = st.select_slider(
                "Overall Mood (right now)",
                options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda x: ['Very Sad', 'Sad', 'Neutral', 'Happy', 'Very Happy'][x - 1]
            )

            st.markdown("---")
            st.markdown("#### EEG Status")
            if st.session_state.eeg_analyzer.is_trained:
                st.markdown(f"""
                <div class="eeg-box">
                    <b>EEG Model Active</b><br>
                    Current EEG Score: <b>{st.session_state.eeg_score:.1%}</b><br>
                    Confidence: <b>{st.session_state.eeg_conf:.1%}</b><br>
                    <small>Go to EEG Setup tab to analyse a new sample</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    EEG model not trained.<br>
                    Go to <b>EEG Setup</b> tab and upload <code>emotions.csv</code>
                    from Kaggle to activate EEG analysis.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🔍 ANALYSE STRESS LEVEL (Full 5-Modality Analysis)",
                     use_container_width=True):
            with st.spinner("Running full multimodal analysis..."):
                text_score, text_features, text_conf = \
                    st.session_state.detector.predict_text_sentiment(text_input)
                survey_score = st.session_state.detector.predict_survey_stress(sleep, workload, mood)
                survey_conf = 1.0
                eeg_score = st.session_state.eeg_score
                eeg_conf = st.session_state.eeg_conf

                overall_stress, overall_conf = st.session_state.detector.calculate_overall_stress(
                    facial_score, voice_score, text_score, survey_score,
                    facial_conf, voice_conf, text_conf, survey_conf,
                    eeg=eeg_score, ec=eeg_conf
                )

                if overall_stress >= 0.7:
                    level = 'High'
                    box_cls = 'danger-box'
                    emoji = '🔴'
                elif overall_stress >= 0.4:
                    level = 'Moderate'
                    box_cls = 'warning-box'
                    emoji = '🟡'
                else:
                    level = 'Low'
                    box_cls = 'success-box'
                    emoji = '🟢'

                st.markdown("---")
                st.markdown("## COMPREHENSIVE RESULTS")
                st.markdown(f"""
                <div class="{box_cls}">
                    <h2>{emoji} {level.upper()} STRESS DETECTED</h2>
                    <h3>Overall Stress: {overall_stress:.1%} | System Confidence: {overall_conf:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                c1.metric("Overall", f"{overall_stress:.1%}")
                c2.metric("Level", level)
                c3.metric("Facial", f"{facial_score:.1%}", delta=f"Conf {facial_conf:.0%}")
                c4.metric("Voice", f"{voice_score:.1%}", delta=f"Conf {voice_conf:.0%}")
                c5.metric("Text", f"{text_score:.1%}", delta=f"Conf {text_conf:.0%}")
                c6.metric("Survey", f"{survey_score:.1%}", delta="Conf 100%")
                c7.metric("EEG", f"{eeg_score:.1%}", delta=f"Conf {eeg_conf:.0%}")

                st.markdown("### Overall Stress Meter")
                st.progress(float(overall_stress))

                if overall_conf < 0.7:
                    st.warning(f"Low confidence ({overall_conf:.1%}) - provide more complete input.")

                st.markdown("### Personalised Recommendations")
                for rec in st.session_state.detector.generate_recommendations(
                        level, facial_score, voice_score, text_score):
                    if rec:
                        st.markdown(f"- {rec}")

                entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'overall_stress': overall_stress,
                    'overall_confidence': overall_conf,
                    'level': level,
                    'breakdown': {
                        'facial': facial_score, 'voice': voice_score,
                        'text': text_score, 'survey': survey_score, 'eeg': eeg_score
                    },
                    'confidences': {
                        'facial': facial_conf, 'voice': voice_conf,
                        'text': text_conf, 'survey': survey_conf, 'eeg': eeg_conf
                    },
                }
                if facial_features:
                    entry['facial_features'] = facial_features
                if voice_features:
                    entry['voice_features'] = voice_features
                if text_features:
                    entry['text_features'] = text_features
                if st.session_state.eeg_features:
                    entry['eeg_features'] = st.session_state.eeg_features

                st.session_state.history.append(entry)
                st.success(f"✅ Assessment saved! Total: {len(st.session_state.history)}")

    # ── TAB 2: HISTORY ──────────────────────────────────────
    with tab2:
        st.markdown("### Assessment History & Analytics")
        if st.session_state.history:
            hist = st.session_state.history
            avg = sum(h['overall_stress'] for h in hist) / len(hist)
            avgc = sum(h['overall_confidence'] for h in hist) / len(hist)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Assessments", len(hist))
            c2.metric("Average Stress", f"{avg:.1%}")
            c3.metric("Average Confidence", f"{avgc:.1%}")
            c4.metric("Latest", f"{hist[-1]['overall_stress']:.1%}")

            lc = sum(1 for h in hist if h['level'] == 'Low')
            mc = sum(1 for h in hist if h['level'] == 'Moderate')
            hc = sum(1 for h in hist if h['level'] == 'High')

            st.markdown("#### Stress Level Distribution")
            d1, d2, d3 = st.columns(3)
            d1.metric("Low", lc)
            d2.metric("Moderate", mc)
            d3.metric("High", hc)

            st.markdown("#### Average Modality Scores")
            af = sum(h['breakdown']['facial'] for h in hist) / len(hist)
            av = sum(h['breakdown']['voice'] for h in hist) / len(hist)
            at = sum(h['breakdown']['text'] for h in hist) / len(hist)
            asv = sum(h['breakdown']['survey'] for h in hist) / len(hist)
            ae = sum(h['breakdown'].get('eeg', 0.5) for h in hist) / len(hist)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Facial", f"{af:.1%}")
            m2.metric("Voice", f"{av:.1%}")
            m3.metric("Text", f"{at:.1%}")
            m4.metric("Survey", f"{asv:.1%}")
            m5.metric("EEG", f"{ae:.1%}")

            st.markdown("#### Recent Assessments")
            for entry in reversed(hist[-10:]):
                with st.expander(
                    f"{entry['timestamp']} - {entry['level']} "
                    f"({entry['overall_stress']:.1%}) - "
                    f"Conf: {entry['overall_confidence']:.1%}"
                ):
                    cols = st.columns(5)
                    cols[0].metric("Facial", f"{entry['breakdown']['facial']:.1%}")
                    cols[1].metric("Voice", f"{entry['breakdown']['voice']:.1%}")
                    cols[2].metric("Text", f"{entry['breakdown']['text']:.1%}")
                    cols[3].metric("Survey", f"{entry['breakdown']['survey']:.1%}")
                    cols[4].metric("EEG", f"{entry['breakdown'].get('eeg', 0.5):.1%}")

            st.markdown("---")
            st.markdown("#### Export Options")
            e1, e2, e3 = st.columns(3)

            with e1:
                csv_rows = []
                for h in hist:
                    csv_rows.append({
                        'timestamp': h['timestamp'],
                        'overall_stress': f"{h['overall_stress']:.4f}",
                        'overall_confidence': f"{h['overall_confidence']:.4f}",
                        'level': h['level'],
                        'facial': f"{h['breakdown']['facial']:.4f}",
                        'voice': f"{h['breakdown']['voice']:.4f}",
                        'text': f"{h['breakdown']['text']:.4f}",
                        'survey': f"{h['breakdown']['survey']:.4f}",
                        'eeg': f"{h['breakdown'].get('eeg', 0.5):.4f}",
                    })
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
                st.download_button("Download CSV",
                                   buf.getvalue(), "neurosense_history.csv", "text/csv",
                                   use_container_width=True)

            with e2:
                json_str = json.dumps(hist, indent=2, default=str)
                st.download_button("Download JSON Backup",
                                   json_str, "neurosense_backup.json", "application/json",
                                   use_container_width=True)

            with e3:
                uploaded_json = st.file_uploader("Upload JSON Backup", type='json', key="json_upload")
                if uploaded_json:
                    try:
                        loaded = json.load(uploaded_json)
                        st.session_state.history = loaded
                        st.success(f"✅ Loaded {len(loaded)} assessments!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            st.markdown("---")
            user_name = st.text_input("Your Name for PDF Report", "Anonymous")
            if st.button("Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_path = generate_pdf_report(user_name, hist)
                    with open(pdf_path, 'rb') as f:
                        st.download_button(
                            "Download PDF",
                            f,
                            file_name=f"neurosense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    os.unlink(pdf_path)
                    st.success("✅ Report generated!")
        else:
            st.warning("No assessments yet. Complete an assessment first!")

    # ── TAB 3: EEG SETUP ────────────────────────────────────
    with tab3:
        st.markdown("### EEG Brain Signal Analysis - Dataset Setup")
        st.markdown("""
        <div class="eeg-box">
            <b>Dataset Required:</b> EEG Dataset for Stress Detection (Kaggle)<br>
            <b>Link:</b> https://www.kaggle.com/datasets/prashantgehlot2404/eeg-dataset-stress-detection<br>
            <b>File to upload:</b> emotions.csv<br>
            <b>Subjects:</b> 40 | <b>Tasks:</b> Stroop, Arithmetic, Mirror Image, Relaxation
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Step 1 - Upload Dataset and Train Model")
        eeg_csv_file = st.file_uploader(
            "Upload emotions.csv",
            type=['csv'],
            key="eeg_upload",
            help="Download from Kaggle and upload the CSV file here."
        )

        if eeg_csv_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(eeg_csv_file.read())
                st.session_state.eeg_csv_path = tmp.name

            if not st.session_state.eeg_analyzer.is_trained:
                with st.spinner("Training EEG Random Forest model... (one-time setup)"):
                    result = st.session_state.eeg_analyzer.load_and_train(
                        st.session_state.eeg_csv_path
                    )
                if result['success']:
                    st.success(
                        f"EEG Model Trained! "
                        f"Train Acc: {result['train_accuracy']:.1%} | "
                        f"Test Acc: {result['test_accuracy']:.1%} | "
                        f"Samples: {result['samples']} | "
                        f"Features: {result['features']}"
                    )
                    st.balloons()
                else:
                    st.error(f"Training failed: {result['error']}")
            else:
                st.info("EEG model already trained this session.")

        st.markdown("---")
        if st.session_state.eeg_analyzer.is_trained:
            st.markdown("#### Step 2 - Analyse an EEG Sample")
            p1, p2 = st.columns(2)

            with p1:
                st.markdown("**Demo Mode - Random Sample from Dataset**")
                if st.button("Analyse Random EEG Sample", use_container_width=True):
                    if st.session_state.eeg_csv_path:
                        s, status_msg, fd, c, actual = \
                            st.session_state.eeg_analyzer.get_random_sample_prediction(
                                st.session_state.eeg_csv_path
                            )
                        st.session_state.eeg_score = s
                        st.session_state.eeg_conf = c
                        st.session_state.eeg_features = fd
                        label_text = "Stressed" if actual != 0 else "Relaxed"
                        st.markdown(f"""
                        <div class="eeg-box">
                            <b>EEG Stress Score: {s:.1%}</b><br>
                            Confidence: {c:.1%}<br>
                            Dataset Label: {label_text}
                        </div>
                        """, unsafe_allow_html=True)
                        st.info(f"{status_msg}")
                        if fd:
                            with st.expander("Detailed EEG Features (17+)"):
                                col_a, col_b = st.columns(2)
                                for i, (k, v) in enumerate(fd.items()):
                                    if i % 2 == 0:
                                        col_a.text(f"{k}: {v:.4f}")
                                    else:
                                        col_b.text(f"{k}: {v:.4f}")

            with p2:
                st.markdown("**Manual Input - Paste EEG Row Values**")
                custom_eeg = st.text_area(
                    "Paste comma-separated EEG values (one epoch):",
                    placeholder="0.12, -0.34, 0.56, 1.23, ...",
                    height=100
                )
                if st.button("Analyse Custom EEG", use_container_width=True):
                    if custom_eeg.strip():
                        try:
                            eeg_vals = np.array([float(v) for v in custom_eeg.split(',')])
                            s, status_msg, fd, c = \
                                st.session_state.eeg_analyzer.predict_eeg_stress(eeg_vals)
                            st.session_state.eeg_score = s
                            st.session_state.eeg_conf = c
                            st.session_state.eeg_features = fd
                            st.markdown(f"""
                            <div class="eeg-box">
                                <b>EEG Stress Score: {s:.1%}</b><br>
                                Confidence: {c:.1%}
                            </div>
                            """, unsafe_allow_html=True)
                            st.info(f"{status_msg}")
                        except Exception as e:
                            st.error(f"Invalid input: {e}")
                    else:
                        st.warning("Please paste EEG values first.")

            st.markdown("---")
            st.markdown("#### Current EEG Reading (used in next assessment)")
            ea, eb = st.columns(2)
            ea.metric("EEG Stress Score", f"{st.session_state.eeg_score:.1%}")
            eb.metric("EEG Confidence", f"{st.session_state.eeg_conf:.1%}")
        else:
            st.info("Upload emotions.csv above to activate EEG analysis.")

        st.markdown("---")
        st.markdown("""
#### About the EEG Dataset
| Property | Value |
|---|---|
| Source | Kaggle - prashantgehlot2404 |
| Subjects | 40 |
| Channels | 32 |
| Sampling Rate | 128 Hz |
| Tasks | Stroop Test, Arithmetic, Mirror Image, Relaxation |
| Labels | Stress / Relaxed (binarised) |
| Classifier | Random Forest (100 trees, max depth 10) |
| Features Extracted | 17+ (band powers, ratios, entropy, Hjorth) |
        """)

    # ── TAB 4: FEATURES ─────────────────────────────────────
    with tab4:
        st.markdown("### Implemented Features (67+)")
        st.markdown("""
#### Facial Expression Analysis (15 Features)
1. Facial Symmetry (left-right asymmetry index)
2. Overall Face Brightness
3. Eye Region Brightness
4. Texture Variance (Laplacian variance)
5. Eye Count (Haar Cascade detection)
6. Overall Edge Density (Canny edges)
7. Forehead Edge Density (wrinkle stress indicator)
8. Contrast Ratio (pixel standard deviation)
9. Normalised Face Size (fraction of frame)
10. RGB Red Channel Mean
11. RGB Green Channel Mean
12. RGB Blue Channel Mean
13. Surface Roughness (texture/brightness ratio)
14. Face Count Confidence
15. Eye Detection Confidence

---

#### Voice Stress Analysis (12 Features)
1. Audio Duration
2. Pitch Detection via Autocorrelation
3. Pitch Mean (average fundamental frequency Hz)
4. Pitch Standard Deviation
5. Pitch Variability Ratio
6. Active Pitch Segments count
7. RMS Energy Mean
8. Energy Standard Deviation
9. Energy Variation Coefficient
10. Zero-Crossing Rate
11. Pause / Silence Detection count
12. Amplitude Dynamics (max, mean, dynamic range)

---

#### Text Sentiment Analysis (13 Features)
1. Weighted Stress Keyword Score (30+ keywords)
2. Weighted Positive Keyword Score (21+ keywords)
3. Negation Handling (context-aware, 3-word window)
4. Character Count
5. Word Count
6. Sentence Count
7. Average Sentence Length
8. Exclamation Mark Frequency
9. Question Mark Frequency
10. Ellipsis Detection
11. Capitalisation Ratio
12. Unique Word Ratio
13. Sentence Complexity Score

---

#### Lifestyle Survey (3 Factors)
1. Sleep Quality (1-5 scale, exponent 1.2)
2. Workload / Academic Pressure (1-5 scale, exponent 1.1)
3. Overall Mood (1-5 scale, exponent 1.2)

---

#### EEG Brain Signal Analysis (17+ Features)
1. Delta Band Power (0.5-4 Hz)
2. Theta Band Power (4-8 Hz)
3. Alpha Band Power (8-13 Hz)
4. Beta Band Power (13-30 Hz)
5. Gamma Band Power (30-45 Hz)
6. Alpha/Beta Ratio (PRIMARY stress biomarker)
7. Theta/Alpha Ratio
8. Spectral Entropy
9. Total Power
10. Signal Mean
11. Signal Standard Deviation
12. Signal Variance
13. Signal Range
14. Skewness
15. Kurtosis
16. Hjorth Activity
17. Hjorth Mobility
18. Hjorth Complexity
19. Random Forest Classifier (40-subject Kaggle dataset)

---

#### Advanced System Features
- 5-Modality Confidence-Weighted Fusion
- Per-modality confidence scoring
- Context-aware personalised recommendations (30+)
- Historical assessment tracking
- CSV export, JSON backup/restore, PDF report generation
- Local processing - No external APIs - Privacy-first
- Crisis helpline integration (India)
        """)

    # ── TAB 5: ABOUT ────────────────────────────────────────
    with tab5:
        st.markdown("""
### About NeuroSense

**AI-Based Multimodal Stress Detection System**
Anurag University | Team 17, IT-D, 4th B.Tech II-SEM
Members: A Raviteja, B Bharatha Ratna, J Nagesh
Mentor: Mr. Mahesh Kumar Swamy

---

#### Technology Stack
| Component | Library |
|---|---|
| Frontend / UI | Streamlit |
| Computer Vision | OpenCV 4.x |
| Signal Processing | NumPy + SciPy |
| EEG ML Model | scikit-learn (Random Forest) |
| NLP Engine | Custom (no external APIs) |
| PDF Reports | FPDF |
| Data Handling | Pandas |

---

#### Five Analysis Modalities
1. Facial - 15 features: symmetry, texture, edges, brightness, eye detection
2. Voice - 12 features: pitch (autocorrelation), energy, ZCR, pauses
3. Text - 13 features: negation-aware NLP, complexity, punctuation
4. Survey - 3 factors: sleep, workload, mood (exponential weighting)
5. EEG - 17+ features: band powers, Alpha/Beta ratio, Hjorth, spectral entropy

---

#### Scientific Basis
- Multimodal fusion improves accuracy by 40% over single-modality methods
- Alpha/Beta EEG ratio is a clinically validated stress biomarker
- Facial symmetry correlates with emotional state changes
- Pitch variation increases under psychological stress
- Text linguistic complexity decreases under cognitive load

---

#### How to Use NeuroSense
1. Face - Click camera, capture photo, instant 15-feature analysis
2. Voice - Record 5-10s speech, pitch/energy/pause analysis
3. Text - Type 50+ words about your emotional state
4. Survey - Slide the 3 wellness sliders
5. EEG - Go to EEG Setup tab, upload Kaggle CSV, analyse sample
6. Analyse - Click the big gold button, get full results and recommendations
7. Track - History tab shows trends, distributions, CSV/JSON/PDF exports

---

Educational tool only - not a clinical diagnostic instrument.
For medical concerns, consult a licensed mental health professional.

Anurag University - Building the future of mental health technology
        """)


if __name__ == "__main__":
    main()