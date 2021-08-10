import os
import MeCab
import fugashi
import sys

# NOTE
# The full version of UniDic requires a separate download step
#pip install fugashi[unidic]
#python -m unidic download

def init_mecab(dicflag, mecab_option=None):
    mecab_option = mecab_option or ""

    if dicflag == 'unidic_lite':
        import unidic_lite
        dic_dir = unidic_lite.DICDIR
    elif dicflag == 'unidic':
        import unidic
        dic_dir = unidic.DICDIR
    elif dicflag == 'ipadic':
        import ipadic
        dic_dir = ipadic.DICDIR
    else:
        raise ValueError("Invalid mecab dic {} is specified, preferred unidic_lite, unidic, ipadic".format(dicflag))

    mecabrc = os.path.join(dic_dir, 'mecabrc')
    if os.path.exists(mecabrc):
        mecab_option = '-d {} -r {} '.format(dic_dir, mecabrc) + mecab_option
    else:
        mecab_option = '-d {}'.format(dic_dir) + mecab_option


    mecab = fugashi.GenericTagger(mecab_option)

    return mecab


def mecab_split(mecab, text):
    tokens = list()
    cursor = 0
    for word in mecab(str(text)):
        token = word.surface
        start = str(text).index(token, cursor)
        end = start + len(token)

        tokens.append(text[start:end])
        cursor = end

    return tokens

debug = False
if debug:
    text = "太郎は二郎が作ったケータイを壊した。"

    mecab_ipa = init_mecab('ipadic', None)
    text_out = mecab_split(mecab_ipa, text)
    print(' '.join(text_out))

    mecab_unidic = init_mecab('unidic', None)
    text_out = mecab_split(mecab_unidic, text)
    print(' '.join(text_out))

    mecab_unidic_lite = init_mecab('unidic_lite', None)
    text_out = mecab_split(mecab_unidic_lite, text)
    print(' '.join(text_out))


# nttreso, unidic-lite; and unidic

def main():
    mecab_unidic_lite = init_mecab('unidic_lite')

    for aline in sys.stdin:
        aline = aline.strip()
        aline_out = mecab_split(mecab_unidic_lite, aline)
        print(' '.join(aline_out))

if __name__ == '__main__':
    main()




