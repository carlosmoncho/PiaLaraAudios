from flask import Flask, render_template, request
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import uuid
from pydub import AudioSegment

app = Flask(__name__)

MODEL_PATH = "model"
rutaAudio = "audio_files/"
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="spanish", task="transcribe")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audios/save-record', methods=['POST'])
def save_record():
    audio_file = request.files['file']
    if not os.path.isdir('audio_files'):
        os.makedirs('audio_files')
    name_audio = str(uuid.uuid4()) + '.wav'

    audio_file.save(os.path.join('audio_files', name_audio))
    song = AudioSegment.from_file(rutaAudio + name_audio)
    song.export(rutaAudio + name_audio, format="wav")
    data, s = librosa.load(rutaAudio + name_audio, sr=16000) 
    input_features = processor(data, sampling_rate=16000, return_tensors="pt").input_features 
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return {'status': 'ok', 'message': transcription}

 

app.run(debug=True)