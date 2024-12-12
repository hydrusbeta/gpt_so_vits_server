import argparse
import base64
import json
import os.path
import subprocess
import tempfile
import traceback

import hay_say_common as hsc
import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common.cache import Stage
from jsonschema import ValidationError

ARCHITECTURE_NAME = 'gpt_so_vits'
ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, ARCHITECTURE_NAME)
TEMP_FILE_EXTENSION = '.wav'

PYTHON_EXECUTABLE = os.path.join(hsc.ROOT_DIR, '.venvs', ARCHITECTURE_NAME, 'bin', 'python')

GPT_WEIGHTS_FILE_EXTENSION = '.ckpt'
SO_VITS_WEIGHTS_FILE_EXTENSION = '.pth'
PRECOMPUTATIONS_FILE_EXTENSION = '.safetensors'

REFERENCE_TEXT_FILENAME = 'reference_text.txt'
TARGET_TEXT_FILENAME = 'target_text.txt'

USE_PRECOMPUTED_EMBEDDING = 'Use Precomputed Embeddings'
USE_REFERENCE_AUDIO = "Use Reference Audio"

SUPPORTED_LANGUAGES_MAP = {
    'Chinese (Mandarin)': '中文',
    'English': '英文',
    'Japanese': '日文',
    'Chinese (Cantonese)': '粤语',
    'Korean': '韩文',
    'Mandarin-English Mix': '中英混合',
    'Japanese-English Mix': '日英混合',
    'Cantonese-English Mix': '粤英混合',
    'Korean-English Mix': '韩英混合',
    'Auto Multilingual': '多语种混合',
    'Auto Multilingual (Cantonese)': '多语种混合(粤语)'
}
CUTTING_STRATEGIES_MAP = {
    "No slicing": "不切",
    "One slice every 4 sentences": "凑四句一切",
    "One slice every 50 characters": "凑50字一切",
    "Slice by Mandarin Chinese punctuation": "按中文句号。切",
    "Slice by English punctuation": "按英文句号.切",
    "Slice by punctuation (any language)": "按标点符号切",
}

app = Flask(__name__)


def get_traits(path_to_python_executable, path_to_safetensors):
    """Returns a list of the available "traits" (i.e. emotions) in the safetensors file. This Flask server does not
    have pytorch installed, but pytorch is required to use safe_open, so this method is designed to remotely run
    commands on the virtual environment for GPT SoVITS (which *does* have pytorch) via the subprocess module."""
    code = [f'import json; '
            f'from safetensors import safe_open; {os.linesep}'
            f'with safe_open("{path_to_safetensors}", framework="pt") as f: '
            f'    traits = set([item.split(".")[0] for item in list(f.keys())]); '
            f'print(json.dumps(sorted(list(traits)))); ']
    output = subprocess.check_output([path_to_python_executable, '-c', *code])
    decoded_output = json.loads(output.decode('utf-8'))
    return decoded_output


def register_methods(cache):
    @app.route('/generate', methods=['POST'])
    def generate() -> (str, int):
        code = 200
        message = ""
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                (user_text, character, reference_audio_hash, reference_text, reference_language, target_language,
                 cutting_strategy, top_k, top_p, temperature, do_not_use_ref, speed, additional_refs, trait,
                 reference_option, output_filename_sans_extension, gpu_id, session_id) = parse_inputs()
                if reference_option == USE_REFERENCE_AUDIO:
                    reference_audio = prepare_reference_audio(reference_audio_hash, tempdir, cache, session_id)
                    reference_text_file = prepare_reference_text(reference_text, tempdir)
                    trait = None
                else:  # USE_PRECOMPUTED_EMBEDDING
                    reference_audio = None
                    reference_text_file = None
                    do_not_use_ref = None
                target_text_file = prepare_target_text(user_text, tempdir)
                execute_program(target_text_file, character, reference_audio, reference_text_file, reference_language,
                                target_language, cutting_strategy, top_k, top_p, temperature, do_not_use_ref, speed,
                                additional_refs, trait, gpu_id, tempdir)
                copy_output(tempdir, output_filename_sans_extension, session_id)
        except BadInputException:
            code = 400
            message = traceback.format_exc()
        except Exception:
            code = 500
            message = ('An error occurred while generating the output: \n' + traceback.format_exc() +
                       '\n\nPayload:\n' + json.dumps(request.json))

        # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
        message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
        response = {
            "message": message
        }

        return json.dumps(response, sort_keys=True, indent=4), code

    @app.route('/gpu-info', methods=['GET'])
    def get_gpu_info():
        return hsc.get_gpu_info_from_another_venv(PYTHON_EXECUTABLE)

    @app.route('/available-traits/<character>', methods=['GET'])
    def get_available_traits(character):
        traits = []
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        precomputations_path = hsc.get_files_with_extension(character_dir, PRECOMPUTATIONS_FILE_EXTENSION)
        if precomputations_path:
            traits = get_traits(PYTHON_EXECUTABLE, precomputations_path[0])
        else:
            # This is not necessarily an error condition. It just means we don't have any precomputations available.
            print(f"no precomputations file with extension {PRECOMPUTATIONS_FILE_EXTENSION} in {character_dir}", flush=True)
        return traits

    schema = {
        'type': 'object',
        'properties': {
            'Inputs': {
                'type': 'object',
                'properties': {
                    'User Text': {'type': 'string'},
                    'User Audio': {'type': ['string', 'null']}
                },
                'required': ['User Text']
            },
            'Options': {
                'type': 'object',
                'properties': {
                    'Character': {'type': 'string'},
                    'Reference Audio': {'type': 'string'},
                    'Reference Text': {'type': 'string'},
                    'Reference Language': {'enum': list(SUPPORTED_LANGUAGES_MAP.keys())},
                    'Target Language': {'enum': list(SUPPORTED_LANGUAGES_MAP.keys())},
                    'Cutting Strategy': {'enum': list(CUTTING_STRATEGIES_MAP.keys())},
                    'Top-K': {'type': 'integer', 'minimum': 1},
                    'Top-P': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'Temperature': {'type': 'number', 'minimum': .00001, 'maximum': 1},
                    'Speed': {'type': 'number', 'minimum': .1, 'maximum': 5.0},
                    'Additional Reference Audios': {'type': 'array', 'items': {'type': 'string'}},
                    'Trait': {'type': 'string'},
                    'Reference Option': {'enum': [USE_PRECOMPUTED_EMBEDDING, USE_REFERENCE_AUDIO]},
                },
                'required': ['Character', 'Reference Language', 'Target Language', 'Cutting Strategy', 'Top-K', 'Top-P',
                             'Temperature', 'Speed', 'Reference Option']
            },
            'Output File': {'type': 'string'},
            'GPU ID': {'type': ['string', 'integer']},
            'Session ID': {'type': ['string', 'null']}
        },
        'required': ['Inputs', 'Options', 'Output File', 'GPU ID', 'Session ID']
    }

    def parse_inputs():
        try:
            jsonschema.validate(instance=request.json, schema=schema)
        except ValidationError as e:
            raise BadInputException(e.message)

        user_text = request.json['Inputs']['User Text']
        character = request.json['Options']['Character']
        reference_audio_hash = request.json['Options'].get('Reference Audio')
        reference_text = request.json['Options'].get('Reference Text')
        reference_language = SUPPORTED_LANGUAGES_MAP[request.json['Options']['Reference Language']]
        target_language = SUPPORTED_LANGUAGES_MAP[request.json['Options']['Target Language']]
        cutting_strategy = CUTTING_STRATEGIES_MAP.get(request.json['Options'].get('Cutting Strategy'))
        top_k = request.json['Options']['Top-K']
        top_p = request.json['Options']['Top-P']
        temperature = request.json['Options']['Temperature']
        do_not_use_ref = not reference_text
        speed = request.json['Options']['Speed']
        additional_refs = request.json['Options'].get('Additional Reference Audios', [])
        trait = request.json['Options'].get('Trait')
        reference_option = request.json['Options']['Reference Option']
        output_filename_sans_extension = request.json['Output File']
        gpu_id = request.json['GPU ID']
        session_id = request.json['Session ID']

        return (user_text, character, reference_audio_hash, reference_text, reference_language, target_language,
                cutting_strategy, top_k, top_p, temperature, do_not_use_ref, speed, additional_refs, trait,
                reference_option, output_filename_sans_extension, gpu_id, session_id)

    class BadInputException(Exception):
        pass

    def prepare_reference_audio(input_hash, tempdir, cache, session_id):
        """Temporarily pull the reference file out of the cache and save it to a file."""
        if input_hash is None:
            return None
        target = os.path.join(tempdir, input_hash + TEMP_FILE_EXTENSION)
        try:
            array, samplerate = cache.read_audio_from_cache(Stage.RAW, session_id, input_hash)
            soundfile.write(target, array, samplerate)
        except Exception as e:
            raise Exception("Unable to save reference audio to a temporary file.") \
                from e
        return target

    def prepare_reference_text(reference_text, tempdir):
        if reference_text is None:
            return None
        target = os.path.join(tempdir, REFERENCE_TEXT_FILENAME)
        with open(target, 'w') as file:
            file.write(reference_text)
        return target

    def prepare_target_text(target_text, tempdir):
        target = os.path.join(tempdir, TARGET_TEXT_FILENAME)
        with open(target, 'w') as file:
            file.write(target_text)
        return target

    def execute_program(target_text_file, character, reference_audio, reference_text_file, reference_language,
                        target_language, cutting_strategy, top_k, top_p, temperature, do_not_use_ref, speed,
                        additional_refs, trait, gpu_id, tempdir):
        character_dir = hsc.character_dir(ARCHITECTURE_NAME, character)
        arguments = [
            '--gpt_model', hsc.get_single_file_with_extension(character_dir, GPT_WEIGHTS_FILE_EXTENSION),
            '--sovits_model', hsc.get_single_file_with_extension(character_dir, SO_VITS_WEIGHTS_FILE_EXTENSION),
            *(['--precomputed_traits_file', hsc.get_single_file_with_extension(character_dir, PRECOMPUTATIONS_FILE_EXTENSION)] if trait is not None else [None, None]),
            *(['--ref_audio', reference_audio] if reference_audio is not None else [None, None]),
            *(['--ref_text', reference_text_file] if reference_text_file is not None else [None, None]),
            '--ref_language', reference_language,
            '--target_text', target_text_file,
            '--target_language', target_language,
            '--output_path', tempdir,
            '--speed', str(speed),
            '--how_to_cut', cutting_strategy,
            '--top_k', str(top_k),
            '--top_p', str(top_p),
            '--temperature', str(temperature),
            *(['--ref_free'] if do_not_use_ref else [None]),
            *(['--precomputed_trait', trait] if trait is not None else [None, None]),
            '--additional_inp_refs', *additional_refs,
        ]
        arguments = [argument for argument in arguments if argument]  # Removes all "None" objects in the list.
        env = hsc.select_hardware(gpu_id)
        subprocess.run([PYTHON_EXECUTABLE, '-m', 'GPT_SoVITS.inference_cli', *arguments], env=env)

    def copy_output(tempdir, output_filename_sans_extension, session_id):
        array_output, sr_output = hsc.read_audio(os.path.join(tempdir, 'output.wav'))
        cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_filename_sans_extension, array_output, sr_output)


def parse_arguments():
    parser = argparse.ArgumentParser(prog='main.py', description='A webservice interface for voice conversion with RVC')
    parser.add_argument('--cache_implementation', default='file', choices=hsc.cache_implementation_map.keys(), help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    cache = hsc.select_cache_implementation(args.cache_implementation)
    register_methods(cache)
    app.run(host='0.0.0.0', port=6581)
