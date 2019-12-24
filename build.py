#!/usr/bin/env python3
import base64
import gzip
import os
from pathlib import Path

is_Windows = True if os.name == "nt" else False


def encode_file(path: Path) -> str:
    compressed = gzip.compress(path.read_bytes(), compresslevel=9)
    return base64.b64encode(compressed).decode('utf-8')


def build_script():
    to_encode = list(Path('titanic_sample').glob('*.py')) + [Path('setup.py')]
    if is_Windows:
        # https://stackoverflow.com/questions/54671385/convert-windowspath-to-posixpath
        file_data = {str(path.as_posix()): encode_file(path) for path in
                     to_encode}
    else:
        file_data = {str(path): encode_file(path) for path in to_encode}
    template = Path('script_template.py').read_text('utf8')
    Path('build/script.py').write_text(
        template.replace('{file_data}', str(file_data)),
        encoding='utf8')


if __name__ == '__main__':
    build_script()
