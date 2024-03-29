# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# copied from fairseq/fairseq/data/indexed_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

# 来自facebook的fairseq的代码

from functools import lru_cache
import os
import shutil
import struct
from itertools import accumulate

import numpy as np
import torch
from megatron import print_rank_0


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16 # 2^16=65536
    else:
        return np.int32

# Python 用下划线作为变量前缀和后缀指定特殊变量
#_xxx 不能用’from module import *’导入
#__xxx__ 系统定义名字
#__xxx 类中的私有变量名
#核心风格：避免用下划线作为变量名的开始。


def get_available_dataset_impl():
    return ['lazy', 'cached', 'mmap']


def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC: # b'TNTIDX\x00\x00', header_magic
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]: # b'MMIDIDX\x00\x00'
                return 'mmap'
            else:
                return None
    else:
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None


def make_builder(out_file, impl, vocab_size=None): 
    # out_file = 'my-gpt2-ja-debug_text_sentence.bin'; impl = 'mmap'; vocab_size = 20573
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))
    else:
        return IndexedDatasetBuilder(out_file)


def make_dataset(path, impl, skip_warmup=False):
    '''path=(例子) fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence.
    Since, there are two files:
    fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence.bin
    fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence.idx

    impl=mmap

    output:
    fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence_train_indexmap_32000000mns_512msl_0.10ssp_1234s.npy
    fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence_valid_indexmap_64320mns_512msl_0.10ssp_1234s.npy
    fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence_test_indexmap_320mns_512msl_0.10ssp_1234s.npy

    '''
    if not IndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None
    if impl == 'infer':
        impl = infer_dataset_impl(path) # 根据path.bin的前8个char来判断impl='cached' or 'mmap'
    if impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        #import pdb; pdb.set_trace()
        return MMapIndexedDataset(path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


def dataset_exists(path, impl):
    if impl == 'mmap':
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16
}


def code(dtype): # dtype = <class 'numpy.uint16'> -> 映射到8, 参考的是dtypes这个词典
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'
    # 例如，从
    # fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence
    # 到 fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence.idx

def data_file_path(prefix_path):
    return prefix_path + '.bin'
    # 例如，从
    # fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence
    # 到 fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence.bin

def create_doc_idx(sizes):
    doc_idx = [0]
    for i, s in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for IndexedDataset"""
    _HDR_MAGIC = b'TNTIDX\x00\x00' # header_magic

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,) # unsigned long long (C) ~ long (python)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack('<QQ', f.read(16))
            self.doc_count = struct.unpack('<Q', f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start]:self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory - 预先载入


class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path):
        super().__init__(path)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx: ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx: ptx + a.size])
            return a
        elif isinstance(idx, slice):
            # Hack just to make this work, can optimizer later if necessary
            sents = []
            for i in range(*idx.indices(len(self))):
                sents.append(self[i])
            return sents


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]
        self.doc_idx = [0]

    def add_item(self, tensor):
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        begin = self.data_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack('<Q', len(self.doc_idx)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream: # idx file
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00' # header_magic

        @classmethod
        def writer(cls, path, dtype):
            '''cls=Index:class, path=index_file, dtype=self._dtype, alike numpy.uint16'''
            '''classmethod修饰符对应的函数不需要实例化，不需要self参数，
            但是第一个参数是表示自身类的cls参数，可以来调用类的属性，类的方法，实例化对象等。
            即，可以直接使用： MMapIndexedDataset.Index.writer(index_file, self._dtype),
            其创建的是_Writer:class的一个对象。
            '''
            class _Writer(object): # 有意思 TODO, 类定义在方法之内！
                def __enter__(self):
                    ''' 在操作文件对象的时候，可以这么写：
                    with open('a.txt') as f:
                        code here
                    上面的写法叫做“上下文管理协议”，即with语句，为了让一个对象兼容with语句，必须在这个对象的类种声明
                    __enter__和__exit__方法！
                    '''
                    self._file = open(path, 'wb') # 'my-gpt2-ja-debug_text_sentence.idx'

                    self._file.write(cls._HDR_MAGIC) # part 1; 9 bytes for b'MMIDIDX\x00\x00', cls here is for Index:class
                    self._file.write(struct.pack('<Q', 1)) # part 2; 8 bytes for '1'(int64)
                    self._file.write(struct.pack('<B', code(dtype))) # 1byte, unsigned char (C), integer (python)
                    # dtype = <class 'numpy.uint16'> -> 8, B=unsignedchar
                    # part 3
                    return self
                    

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize # 2
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers 
                    # pointers = [0, 60, 168, 200, 260, 314] or, 2doc case: pointers = [0, 60, 168, 200, 260, 314, 394, 454]

                def write(self, sizes, doc_idx): # sizes = [30, 54, 16, 30, 27, 40], doc_idx = [0, 6]
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack('<Q', len(sizes))) # 6, 写到index文件; part 4, 所有文档的句子的总和
                    self._file.write(struct.pack('<Q', len(doc_idx))) # 2; part 5, 文档的数量=doc_idx-1

                    #import ipdb; ipdb.set_trace()
                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C')) # part 6, 每个句子中word piece (token)的数量
                    del sizes
                    #import ipdb; ipdb.set_trace()
                    pointers = np.array(pointers, dtype=np.int64) # pointers = [0, 60, 168, 200, 260, 314]
                    self._file.write(pointers.tobytes(order='C')) # part 7, 句子的pointers
                    del pointers

                    #import ipdb; ipdb.set_trace()
                    doc_idx = np.array(doc_idx, dtype=np.int64) # doc_idx = [0, 6]
                    self._file.write(doc_idx.tobytes(order='C')) # part 8, 每个document的起始位置?

                def __exit__(self, exc_type, exc_val, exc_tb): # None, None, None
                    ''' 和__enter__对照 '''
                    self._file.close()

            return _Writer() # 返回的是内部类_Writer的一个对象

        def __init__(self, path, skip_warmup=False): # class Index的构造函数
            # index_file_path(self._path) -> 传输进来的是path.idx文件：
            with open(path, 'rb') as stream:
                magic_test = stream.read(9) # read part 1, 9 byptes
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8)) # 读取8个bytes，Q=unsigned long long(C语言)->long(python)
                # read part 2, with 8 bytes
                assert (1,) == version # attention, TODO, this is (1,), not 1! a tuple now

                dtype_code, = struct.unpack('<B', stream.read(1)) # B=unsigned char (C) -> integer (python)
                # read part 3, with 1 bypte

                self._dtype = dtypes[dtype_code] # 1到8，分别对应不同的数据类型
                self._dtype_size = self._dtype().itemsize # =2, not used

                self._len = struct.unpack('<Q', stream.read(8))[0] # 单个元素的数组？
                # read part 4, with 8 bytes, len(sizes)，所有文档中所有句子的个数

                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                # read part 5, with 8 bytes, len(doc_idx)，文档的个数+1

                offset = stream.tell() # e.g., 34,  返回文件的当前位置，即文件指针当前位置

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer( # frombuffer = 将缓冲区解释为一维数组； read part 6; for example, self._sizes = array([ 26, 284,  24, ...,  59,  23,  25], dtype=int32)
                self._bin_buffer,
                dtype=np.int32,
                count=self._len, # 所有文档中所有句子的个数, e.g., 1686984 sentences in 413854 documents
                offset=offset)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            # 所有句子的pointer; read part 7; e.g., self._sizes=1,686,984; self._sizes.nbyptes=6,747,936 = 1,686,984 * 4 (one int32 with 8 bytes!)

            # nbytes=只是存储数据所占的字节bytes个数
            # a = np.array([[2, 11]], dtype=np.int64)
            # print(a.nbytes) -> 16, 因为int64占8个bytes，这样的话，数据区就是2*8=16了。

            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count, # self._doc_count=413854
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)
            # read part 8, 文档的index，例如[0, 6, 8] _doc_count=真实的文档数量+1; or, e.g., (offset=34; self._sizes.nbytes=6,747,936; self._pointers.nbytes=13,495,872=len(self._sizes) * 8 since it is np.int64 = 8 bytes for one int)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False): # MMapIndexedDataset的构造函数
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        # path = path without .bin or .idx, i.e., path+'.bin' -> bin file, path+'.idx' -> idx file
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup) # megatron.data.indexed_dataset.MMapIndexedDataset.Index object

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")

        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        # 'r' = read-only; 'C' = column-major; cpu-memory
        # 内存映像文件是一种将磁盘上的非常大的二进制数据文件当做内存中的数组进行处理的方式。
        # NumPy实现了一个类似于ndarray的memmap对象，它允许将大文件分成小段进行读写，而不是一次性将整个数组读入内存。
        # memmap也拥有跟普通数组一样的方法，因此，基本上只要是能用于ndarray的算法就也能用于memmap。

        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        # python内置函数
        # 返回给定参数的“内存查看对象”(memory view)
        # 是指对支持缓冲区协议的数据进行包装，在不需要复制对象的基础上允许python代码访问。

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                 count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )


class MMapIndexedDatasetBuilder(object): # MMap = memory map indexed data file (for .bin file = 数据文件相关的)
    def __init__(self, out_file, dtype=np.int64): # out_file = 'my-gpt2-ja-debug_text_sentence.bin'; dtype = <class 'numpy.uint16'>
        self._data_file = open(out_file, 'wb') # write, binary file
        self._dtype = dtype
        self._sizes = [] # 每个句子中word piece的个数
        self._doc_idx = [0] # 主动增加一个0，之后是每个文档中的句子的个数。例子： self._sizes=[30, 54, 16, 30, 27, 40], self._doc_idx=[0, 6]

    def add_item(self, tensor): # tensor = tensor of one sentence
        np_array = np.array(tensor.numpy(), dtype=self._dtype) 
        # np_array = array([20298, 20262,  5049, 20263,   211, 16050,   529,    19,   537,       
        # 16011,  5156,  7684, 15402, 20076,  6561, 20077, 16011,  6408,         
        # 406, 17450, 20257,   214,  8413,  1007, 18073,  8184,   813,         
        # 406,     3, 20258], dtype=uint16)
        # 不知不觉的时候，就已经转换为了id! (from text to id) TODO/okay -> encoded_docs = pool.imap(encoder.encode, fin, 25) 是在这里实现的！

        self._data_file.write(np_array.tobytes(order='C')) # C-order is by default
        self._sizes.append(np_array.size) # 30

    def end_document(self):
        self._doc_idx.append(len(self._sizes)) # [30, 54, 16, 30, 27, 40,    30, 3] 
        # 每个数字代表了一个句子中bpe(word piece)的个数，不同的文档会顺次被追加

    def merge_file_(self, another_file): # TODO a way to speed up the data preparing progress?
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        # another_file传入的也是.bin和.idx的前缀
        assert index.dtype == self._dtype # 两个文件的数据类型要求一致！

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f: # 打开另外一个文件的.bin，
            shutil.copyfileobj(f, self._data_file) # 把f的内容复制给self._data_file (fsrc, fdst)
        
        # TODO this is problemtic - 是否有其他需要整合的信息？
        # .bin文件可以直接追加，因为它是一个个的句子组成的。
        # .idx比较麻烦：
        # 前三个parts不变：
        # idx part 1: 9 bytes, _HDR_MAGIC
        # idx part 2: 8 bytes, 1
        # idx part 3: 1 byte, value=8, for 'numpy.uint16', (vocab id)

        # idx part 4: 8 bytes, len(sizes)=总体的句子的个数 (doc1's num1 + doc2's num2 + ... + docn's numn) -> 这个需要两个idx的part 4相加
        # idx part 5: 8 bytes, len(doc_idx)=总体的文档的个数+1 -> 这个需要两个idx的part 5 相加之后-1 = total doc's num + 1
        # idx part 6: 写sizes，即每个句子的长度(word piece的个数)，这个两个idx的这个部分直接叠加就好了
        # idx part 7: 写pointers，这个的话，第二个idx文件的所有pointers都需要重新计算的！然后追加到第一个idx的最后边就可以了；-> 具体为，pointers2每个元素+= sum(sizes1) * pointers1[1]/sizes1[0]
        # idx part 8: 写_doc_idx，这个也是需要修改第二个idx文件的所有_doc_idx，然后追加到第一个idx的最后边就可以了。-> 具体为，_doc_idx2的第0号元素0被删除，剩余每个元素都+= doc_idx1[-1]; 更多细节参考知乎：https://zhuanlan.zhihu.com/p/388830967/

    def finalize(self, index_file): # e.g., index_file = 'my-gpt2-ja-debug_text_sentence.idx'
        self._data_file.close()
        # 构造一个writer对象：
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index: 
            # index_file='my-gpt2-ja-debug_text_sentence.idx', self._dtype=<class 'numpy.uint16'>
            index.write(self._sizes, self._doc_idx) # self._sizes=[30, 54, 16, 30, 27, 40], self._doc_idx=[0, 6]
