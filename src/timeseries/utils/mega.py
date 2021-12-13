import base64
import binascii
# from crypto import (base64_to_a32, base64_url_decode,
#                     decrypt_attr, a32_to_str, get_chunks, str_to_a32)
#
#
# from Crypto.Cipher import AES
import json
import logging
import os
import random
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from tenacity import retry, wait_exponential, retry_if_exception_type
import requests
from Crypto.Cipher import AES
from Crypto.Util import Counter
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Python3 compatibility
if sys.version_info < (3, ):

    def makebyte(x):
        return x

    def makestring(x):
        return x
else:
    import codecs

    def makebyte(x):
        return codecs.latin_1_encode(x)[0]

    def makestring(x):
        return codecs.latin_1_decode(x)[0]


def aes_cbc_encrypt(data, key):
    aes_cipher = AES.new(key, AES.MODE_CBC, makebyte('\0' * 16))
    return aes_cipher.encrypt(data)


def aes_cbc_decrypt(data, key):
    aes_cipher = AES.new(key, AES.MODE_CBC, makebyte('\0' * 16))
    return aes_cipher.decrypt(data)


def aes_cbc_encrypt_a32(data, key):
    return str_to_a32(aes_cbc_encrypt(a32_to_str(data), a32_to_str(key)))


def aes_cbc_decrypt_a32(data, key):
    return str_to_a32(aes_cbc_decrypt(a32_to_str(data), a32_to_str(key)))


def stringhash(str, aeskey):
    s32 = str_to_a32(str)
    h32 = [0, 0, 0, 0]
    for i in range(len(s32)):
        h32[i % 4] ^= s32[i]
    for r in range(0x4000):
        h32 = aes_cbc_encrypt_a32(h32, aeskey)
    return a32_to_base64((h32[0], h32[2]))


def prepare_key(arr):
    pkey = [0x93C467E3, 0x7DB0C7A4, 0xD1BE3F81, 0x0152CB56]
    for r in range(0x10000):
        for j in range(0, len(arr), 4):
            key = [0, 0, 0, 0]
            for i in range(4):
                if i + j < len(arr):
                    key[i] = arr[i + j]
            pkey = aes_cbc_encrypt_a32(pkey, key)
    return pkey


def encrypt_key(a, key):
    return sum((aes_cbc_encrypt_a32(a[i:i + 4], key)
                for i in range(0, len(a), 4)), ())


def decrypt_key(a, key):
    return sum((aes_cbc_decrypt_a32(a[i:i + 4], key)
                for i in range(0, len(a), 4)), ())


def encrypt_attr(attr, key):
    attr = makebyte('MEGA' + json.dumps(attr))
    if len(attr) % 16:
        attr += b'\0' * (16 - len(attr) % 16)
    return aes_cbc_encrypt(attr, a32_to_str(key))


def decrypt_attr(attr, key):
    attr = aes_cbc_decrypt(attr, a32_to_str(key))
    attr = makestring(attr)
    attr = attr.rstrip('\0')
    return json.loads(attr[4:]) if attr[:6] == 'MEGA{"' else False


def a32_to_str(a):
    return struct.pack('>%dI' % len(a), *a)


def str_to_a32(b):
    if isinstance(b, str):
        b = makebyte(b)
    if len(b) % 4:
        # pad to multiple of 4
        b += b'\0' * (4 - len(b) % 4)
    return struct.unpack('>%dI' % (len(b) / 4), b)


def mpi_to_int(s):
    """
    A Multi-precision integer is encoded as a series of bytes in big-endian
    order. The first two bytes are a header which tell the number of bits in
    the integer. The rest of the bytes are the integer.
    """
    return int(binascii.hexlify(s[2:]), 16)


def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)


def modular_inverse(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m


def base64_url_decode(data):
    data += '=='[(2 - len(data) * 3) % 4:]
    for search, replace in (('-', '+'), ('_', '/'), (',', '')):
        data = data.replace(search, replace)
    return base64.b64decode(data)


def base64_to_a32(s):
    return str_to_a32(base64_url_decode(s))


def base64_url_encode(data):
    data = base64.b64encode(data)
    data = makestring(data)
    for search, replace in (('+', '-'), ('/', '_'), ('=', '')):
        data = data.replace(search, replace)
    return data


def a32_to_base64(a):
    return base64_url_encode(a32_to_str(a))


def get_chunks(size):
    p = 0
    s = 0x20000
    while p + s < size:
        yield (p, s)
        p += s
        if s < 0x100000:
            s += 0x20000
    yield (p, size - p)


def make_id(length):
    text = ''
    possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    for i in range(length):
        text += random.choice(possible)
    return text


class ValidationError(Exception):
    """
    Error in validation stage
    """
    pass


_CODE_TO_DESCRIPTIONS = {
    -1: ('EINTERNAL',
         ('An internal error has occurred. Please submit a bug report, '
          'detailing the exact circumstances in which this error occurred')),
    -2: ('EARGS', 'You have passed invalid arguments to this command'),
    -3: ('EAGAIN',
         ('(always at the request level) A temporary congestion or server '
          'malfunction prevented your request from being processed. '
          'No data was altered. Retry. Retries must be spaced with '
          'exponential backoff')),
    -4: ('ERATELIMIT',
         ('You have exceeded your command weight per time quota. Please '
          'wait a few seconds, then try again (this should never happen '
          'in sane real-life applications)')),
    -5: ('EFAILED', 'The upload failed. Please restart it from scratch'),
    -6:
    ('ETOOMANY',
     'Too many concurrent IP addresses are accessing this upload target URL'),
    -7:
    ('ERANGE', ('The upload file packet is out of range or not starting and '
                'ending on a chunk boundary')),
    -8: ('EEXPIRED',
         ('The upload target URL you are trying to access has expired. '
          'Please request a fresh one')),
    -9: ('ENOENT', 'Object (typically, node or user) not found'),
    -10: ('ECIRCULAR', 'Circular linkage attempted'),
    -11: ('EACCESS',
          'Access violation (e.g., trying to write to a read-only share)'),
    -12: ('EEXIST', 'Trying to create an object that already exists'),
    -13: ('EINCOMPLETE', 'Trying to access an incomplete resource'),
    -14: ('EKEY', 'A decryption operation failed (never returned by the API)'),
    -15: ('ESID', 'Invalid or expired user session, please relogin'),
    -16: ('EBLOCKED', 'User blocked'),
    -17: ('EOVERQUOTA', 'Request over quota'),
    -18: ('ETEMPUNAVAIL',
          'Resource temporarily not available, please try again later'),
    -19: ('ETOOMANYCONNECTIONS', 'many connections on this resource'),
    -20: ('EWRITE', 'Write failed'),
    -21: ('EREAD', 'Read failed'),
    -22: ('EAPPKEY', 'Invalid application key; request not processed'),
}


class RequestError(Exception):
    """
    Error in API request
    """
    def __init__(self, message):
        code = message
        self.code = code
        code_desc, long_desc = _CODE_TO_DESCRIPTIONS[code]
        self.message = f'{code_desc}, {long_desc}'

    def __str__(self):
        return self.message



class Mega:
    def __init__(self, options=None):
        self.schema = 'https'
        self.domain = 'mega.co.nz'
        self.timeout = 160  # max secs to wait for resp from api requests
        self.sid = None
        self.sequence_num = random.randint(0, 0xFFFFFFFF)
        # self.request_id = make_id(10)
        self._trash_folder_node_id = None

        if options is None:
            options = {}
        self.options = options

    @retry(retry=retry_if_exception_type(RuntimeError),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def _api_request(self, data):
        params = {'id': self.sequence_num}
        self.sequence_num += 1

        if self.sid:
            params.update({'sid': self.sid})

        # ensure input data is a list
        if not isinstance(data, list):
            data = [data]

        url = f'{self.schema}://g.api.{self.domain}/cs'
        response = requests.post(
            url,
            params=params,
            data=json.dumps(data),
            timeout=self.timeout,
        )
        json_resp = json.loads(response.text)
        try:
            if isinstance(json_resp, list):
                int_resp = json_resp[0] if isinstance(json_resp[0],
                                                      int) else None
            elif isinstance(json_resp, int):
                int_resp = json_resp
        except IndexError:
            int_resp = None
        if int_resp is not None:
            if int_resp == 0:
                return int_resp
            if int_resp == -3:
                msg = 'Request failed, retrying'
                logger.info(msg)
                raise RuntimeError(msg)
            raise RequestError(int_resp)
        return json_resp[0]

    def _parse_url(self, url):
        """Parse file id and key from url."""
        if '/file/' in url:
            # V2 URL structure
            url = url.replace(' ', '')
            file_id = re.findall(r'\W\w\w\w\w\w\w\w\w\W', url)[0][1:-1]
            id_index = re.search(file_id, url).end()
            key = url[id_index + 1:]
            return f'{file_id}!{key}'
        elif '!' in url:
            # V1 URL structure
            match = re.findall(r'/#!(.*)', url)
            path = match[0]
            return path
        else:
            raise RequestError('Url key missing')


    def download_url(self, url, dest_path=None, dest_filename=None):
        """
        Download a file by it's public url
        """
        path = self._parse_url(url).split('!')
        file_id = path[0]
        file_key = path[1]
        return self._download_file(
            file_handle=file_id,
            file_key=file_key,
            dest_path=dest_path,
            dest_filename=dest_filename,
            is_public=True,
        )

    def _download_file(self,
                       file_handle,
                       file_key,
                       dest_path=None,
                       dest_filename=None,
                       is_public=False,
                       file=None):
        if file is None:
            if is_public:
                file_key = base64_to_a32(file_key)
                file_data = self._api_request({
                    'a': 'g',
                    'g': 1,
                    'p': file_handle
                })
            else:
                file_data = self._api_request({
                    'a': 'g',
                    'g': 1,
                    'n': file_handle
                })

            k = (file_key[0] ^ file_key[4], file_key[1] ^ file_key[5],
                 file_key[2] ^ file_key[6], file_key[3] ^ file_key[7])
            iv = file_key[4:6] + (0, 0)
            meta_mac = file_key[6:8]
        else:
            file_data = self._api_request({'a': 'g', 'g': 1, 'n': file['h']})
            k = file['k']
            iv = file['iv']
            meta_mac = file['meta_mac']

        # Seems to happens sometime... When this occurs, files are
        # inaccessible also in the official also in the official web app.
        # Strangely, files can come back later.
        if 'g' not in file_data:
            raise RequestError('File not accessible anymore')
        file_url = file_data['g']
        file_size = file_data['s']
        attribs = base64_url_decode(file_data['at'])
        attribs = decrypt_attr(attribs, k)

        if dest_filename is not None:
            file_name = dest_filename
        else:
            file_name = attribs['n']

        input_file = requests.get(file_url, stream=True).raw

        if dest_path is None:
            dest_path = ''
        else:
            dest_path += '/'

        with tempfile.NamedTemporaryFile(mode='w+b',
                                         prefix='megapy_',
                                         delete=False) as temp_output_file:
            k_str = a32_to_str(k)
            counter = Counter.new(128,
                                  initial_value=((iv[0] << 32) + iv[1]) << 64)
            aes = AES.new(k_str, AES.MODE_CTR, counter=counter)

            mac_str = '\0' * 16
            mac_encryptor = AES.new(k_str, AES.MODE_CBC,
                                    mac_str.encode("utf8"))
            iv_str = a32_to_str([iv[0], iv[1], iv[0], iv[1]])

            for chunk_start, chunk_size in tqdm(get_chunks(file_size)):
                chunk = input_file.read(chunk_size)
                chunk = aes.decrypt(chunk)
                temp_output_file.write(chunk)

                encryptor = AES.new(k_str, AES.MODE_CBC, iv_str)
                for i in range(0, len(chunk) - 16, 16):
                    block = chunk[i:i + 16]
                    encryptor.encrypt(block)

                # fix for files under 16 bytes failing
                if file_size > 16:
                    i += 16
                else:
                    i = 0

                block = chunk[i:i + 16]
                if len(block) % 16:
                    block += b'\0' * (16 - (len(block) % 16))
                mac_str = mac_encryptor.encrypt(encryptor.encrypt(block))

                file_info = os.stat(temp_output_file.name)

            file_mac = str_to_a32(mac_str)
            # check mac integrity
            if (file_mac[0] ^ file_mac[1],
                    file_mac[2] ^ file_mac[3]) != meta_mac:
                raise ValueError('Mismatched mac')
            output_path = Path(dest_path + file_name)
            temp_output_file.close()
            shutil.move(temp_output_file.name, output_path)
            return output_path