import streamlit as st
import whisper
import os
import tempfile
import io

st.set_page_config(
    page_title="Bisaya Transcriber",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Bisaya Audio Transcriber")
st.markdown(
    "Upload an audio file to transcribe Bisaya/Cebuano speech using OpenAI Whisper. "
    "Outputs a plain-text transcript and an `.srt` subtitle file."
)

# ── Sidebar: model & settings ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_size = st.selectbox(
        "Whisper model",
        ["large-v3-turbo", "large-v3", "medium", "small", "base"],
        index=0,
        help="Larger models are more accurate but slower."
    )
    language = st.selectbox(
        "Base language hint",
        ["tl", "ceb", "en"],
        index=0,
        help="'tl' (Tagalog) captures Bisaya phonetics well. 'ceb' is Cebuano."
    )
    fp16 = st.checkbox("Use FP16 (GPU only)", value=False)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05,
                            help="0 = strict/deterministic, higher = more creative")
    st.markdown("---")
    st.caption("Built from VER2_TRANSCRIBE.ipynb")

# ── Bisaya vocabulary prompt (from notebook) ────────────────────────────────
BISAYA_PROMPT = (
    "GenAI hmmm uhmm man unsay tawag ana? nag-generate ug mag-use ug dili gyud dili jud "
    "nimo naa mi naa daw imo sundon ako-ako ra mag-generate naa ba ka'y i-try nga "
    "Puwede ra mag bisaya noh? noh? unsaon ingana kana kanang accads "
    "unsa asa diin kaon tulog balay kana diri didto kinsa "
    "unsay pangalan ako ikaw siya kami kamo sila ana ani adto "
    "mao na dili oo sigurado lang gud kay basin unsaon nako "
    "unsa may ato niana niini karon unya gahapon ugma "
    "tagalog english bisaya cebuano interview discussion talk "
    "kuan kuwan koan kwan kuan-ra kuan-ang kana-bang kana-ganing "
    "ba gud gayud gyud lang pa na man diay bitaw jud ra siguro "
    "kaning kaniya kaito ani ana adto karon niini niana niadto kato kanang kaniadto "
    "eskwelahan klase leksyon assignment project exam quiz paper essay "
    "report presentation groupwork major minor subject course professor "
    "teacher classmate schoolwork sulat basa review study search cite "
    "reference paraphrase summarize explain understand lisod dali tulong "
    "tabang pasabot meaning pananglitan "
    "magparaphrase magresearch magcite naggenerate giparaphrase gisummarize "
    "gi-generate gi-cite gi-finish mo-submit pag-search "
    "kanang-the kanang-essay kanang-reference kanang-AI kanang-study "
    "na-generate na-cite na-finish na-submit ang-essay ang-reference ang-AI "
    "unsaon-nako paano giunsa kanus-a diin kinsa ngano unsa-may asa-man "
    "basin siguro tingali mahimo pwede okay-ra walay-problema sige sige-ra "
    "una sunod lastly importante siguroha pasabot meaning "
    "in-other-words sama-sa pananglitan for-example actually honestly "
    "to-be-honest by-the-way speaking-of-which regarding about "
    "uh um ah hmm like you-know I-mean actually wait let-me-see "
    "paano-ba unsaon-diay wait-lang hmmm tagal ah okay sige "
    "permission-to-record can-you-tell-me your-name program year-level school "
    "when-was-the-last-time what-was-that can-you-tell-me-about what-made-you-decide "
    "do-you-use other-than except for-example so-yeah basically just "
    "kay so tapos then pero but ug and kung if kapan when"
)


def format_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def build_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_time(seg['start'])} --> {format_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


@st.cache_resource(show_spinner="Loading Whisper model…")
def load_model(size: str):
    return whisper.load_model(size)


# ── File uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["mp3", "mp4", "wav", "m4a", "ogg", "flac", "webm"],
    help="Supported formats: MP3, MP4, WAV, M4A, OGG, FLAC, WEBM"
)

if uploaded_file:
    st.audio(uploaded_file, format=uploaded_file.type)
    base_name = os.path.splitext(uploaded_file.name)[0]

    if st.button("🚀 Transcribe", type="primary", use_container_width=True):

        # Save upload to a temp file so Whisper can read it
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Loading model…"):
                model = load_model(model_size)

            progress = st.progress(0, text="Transcribing…")

            with st.spinner("Transcribing audio — this may take a while…"):
                result = model.transcribe(
                    tmp_path,
                    language=language,
                    verbose=False,
                    fp16=fp16,
                    temperature=temperature,
                    initial_prompt=BISAYA_PROMPT,
                    word_timestamps=True,
                    compression_ratio_threshold=2.5,
                    condition_on_previous_text=False,
                )

            progress.progress(100, text="Done!")

            transcript_text = result["text"]
            srt_text = build_srt(result["segments"])

            # ── Results ────────────────────────────────────────────────────
            st.success("✅ Transcription complete!")

            tab1, tab2 = st.tabs(["📄 Transcript", "🕐 Subtitles (SRT)"])

            with tab1:
                st.text_area("Full transcript", transcript_text, height=300)
                st.download_button(
                    "⬇️ Download .txt",
                    data=transcript_text.encode("utf-8"),
                    file_name=f"{base_name}_transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with tab2:
                st.text_area("SRT subtitles", srt_text, height=300)
                st.download_button(
                    "⬇️ Download .srt",
                    data=srt_text.encode("utf-8"),
                    file_name=f"{base_name}_subtitles.srt",
                    mime="text/plain",
                    use_container_width=True,
                )

            # Word-level confidence summary
            with st.expander("📊 Word-level details"):
                word_rows = []
                for seg in result["segments"]:
                    for w in seg.get("words", []):
                        word_rows.append({
                            "Word": w["word"].strip(),
                            "Start (s)": round(w["start"], 2),
                            "End (s)": round(w["end"], 2),
                            "Confidence": round(w.get("probability", 0), 3),
                        })
                if word_rows:
                    st.dataframe(word_rows, use_container_width=True)

        finally:
            os.unlink(tmp_path)

else:
    st.info("👆 Upload an audio file to get started.")

st.markdown("---")
st.caption("Powered by [OpenAI Whisper](https://github.com/openai/whisper) · Model: `large-v3-turbo` by default")
