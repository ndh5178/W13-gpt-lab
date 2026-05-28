# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""
import json
from pathlib import Path
import json

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens()

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for i, token in enumerate(SPECIAL_TOKENS):  #enumerate -> 리스트에서 값을 꺼낼 때 번호표를 붙여서 꺼내는 매서드
            self.id_to_token[i] = token
            self.token_to_id[token] = i

        for byte_value in range(NUM_BYTES):
            self.id_to_token[BYTE_OFFSET + byte_value] = bytes([byte_value])
            self.token_to_id[bytes([byte_value])] = BYTE_OFFSET + byte_value

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        id_corpus=[]
        
        for token in list(corpus.encode("utf-8")):
            id_corpus.append(self.token_to_id[bytes([token])])
        
        while len(self.id_to_token) < self.vocab_size:
            count={}

            for i in range(len(id_corpus)-1):
                pair = (id_corpus[i],id_corpus[i+1])
                if pair not in count:
                    count[pair] = 0
                count[pair]+=1    
            
            if not count:
                break

            max_pair = max(count, key=count.get)
            new_id = len(self.id_to_token)
            self.token_to_id[max_pair]=new_id
            self.id_to_token[new_id]=max_pair
            self.merges.append(max_pair)

            i = 0
            while i < len(id_corpus):
                if i < len(id_corpus) - 1 and (id_corpus[i], id_corpus[i + 1]) == max_pair:
                    id_corpus[i:i + 2] = [new_id]
                    i += 1
                else:
                    i += 1

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        path=Path(path)
        data_id=[]
        data_merge=[]
        for id,token in self.id_to_token.items():
            if type(token) == bytes:
                item = {
                    "id": id,
                    "type": "bytes",
                    "value": list(token),
                }
            elif type(token) == tuple:
                item = {
                    "id": id,
                    "type": "tuple",
                    "value": list(token),
                }
            elif type(token) == str:
                item = {
                    "id": id,
                    "type": "str",
                    "value": token,
                }
            data_id.append(item)
        
        for i in self.merges:
            data_merge.append(list(i))
        
        data = {
            "vocab_size": self.vocab_size,
            "id": data_id,
            "merges": data_merge
        }
        
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        for item in data["id"]:
            id = item["id"]
            type_ = item["type"]
            value = item["value"]
            if type_ == "bytes":
                self.id_to_token[id] = bytes(value)
                self.token_to_id[bytes(value)] = id
            elif type_ == "tuple":
                self.id_to_token[id] = tuple(value)
                self.token_to_id[tuple(value)] = id
            elif type_ == "str":
                self.id_to_token[id] = value
                self.token_to_id[value] = id

        for i in data["merges"]:
            self.merges.append(tuple(i))
        


    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        text_byte=[]
        for token in list(text.encode("utf-8")):
            text_byte.append(self.token_to_id[bytes([token])])
             
        for rule in self.merges:
            i=0
            while i<len(text_byte):
                if i < len(text_byte) - 1 and (text_byte[i], text_byte[i + 1]) == rule:
                    text_byte[i:i + 2] = [self.token_to_id[(text_byte[i], text_byte[i + 1])]]
                    i+=1
                else:
                    i+=1
        
        if add_bos_eos:
            text_byte = [self.token_to_id["<bos>"]] + text_byte + [self.token_to_id["<eos>"]]
        
        return text_byte
    
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        skip_ids=[]
        for token in ids:
            if skip_special and token in SPECIAL_IDS.values():
                continue
            skip_ids.append(token)
        
        i=0
        while i < len(skip_ids):
            if skip_ids[i]>259:
                origin_token=self.id_to_token[skip_ids[i]]
                skip_ids[i:i+1]=origin_token
            else:
                i+=1
        
        for j in range(len(skip_ids)):
            skip_ids[j]-=4

        return bytes(skip_ids).decode("utf-8")
