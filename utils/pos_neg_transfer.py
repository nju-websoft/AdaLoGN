from operator import itemgetter

import stanza
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


def get_third_singular_verb(verb):
    if verb.endswith('y'):
        verb = verb[:-1] + 'ies'
    elif verb.endswith('ch') or verb.endswith('s') or verb.endswith('sh') or verb.endswith(
            'x') or verb.endswith('z'):
        verb += 'es'
    elif verb == 'have':
        verb = 'has'
    else:
        verb += 's'
    return verb


def get_feature(feat_str, feat, feat_val) -> bool:
    if feat == '_':
        return False
    feats = dict([tuple(f.split('=')) for f in feat_str.split('|')])
    if feat not in feats:
        return False
    return feats[feat] == feat_val


def antonyms_for(word):
    antonyms = []
    for ss in wn.synsets(word):
        for lemma in ss.lemmas():
            any_pos_antonyms = [antonym.name() for antonym in lemma.antonyms()]
            for antonym in any_pos_antonyms:
                antonym_synsets = wn.synsets(antonym)
                if wn.ADV not in [ss.pos() for ss in antonym_synsets]:
                    continue
                antonyms.append(antonym)
    return list(set(antonyms))


def convert(pos, return_adv_neg=False):
    global v_index

    for v_index in range(len(pos)):
        if pos[v_index][1] != 'VERB':
            continue
        if return_adv_neg:
            ret_list = []
            for index2 in range(len(pos)):
                if pos[index2][1] == 'ADV':
                    antonyms = antonyms_for(pos[index2][0])
                    for antonym in antonyms:
                        ret_list.append(
                            ' '.join(list(map(itemgetter(0), pos[:index2]))) + ' ' + antonym + ' ' + ' '.join(
                                list(map(itemgetter(0), pos[index2 + 1:]))))
            return ret_list
    if return_adv_neg:
        return []

    for v_index in range(len(pos)):
        # if pos[v_index][1] == 'VERB' and get_feature(pos[v_index][2], 'VerbForm', 'Part'):
        #    continue

        if pos[v_index][1] == 'VERB' and pos[v_index][3] != 0:
            continue

        if pos[v_index][1] in ['AUX', 'VERB']:
            if pos[v_index][1] == 'AUX':
                if pos[pos[v_index][3] - 1][3] != 0:
                    continue
            break

    # print(pos[v_index][0])

    # print(pos)
    if v_index + 2 < len(pos) and (pos[v_index + 1][0] == 'not' or pos[v_index + 1][0] == 'n\'t') and pos[v_index + 1][
        1] == 'PART':  # 助动词后面跟否定词
        if pos[v_index][1] != 'AUX' and pos[v_index][1] != 'VERB':
            print(pos)
        assert pos[v_index][1] == 'AUX' or pos[v_index][1] == 'VERB'

        if WordNetLemmatizer().lemmatize(pos[v_index][0], 'v') == 'be':
            return ' '.join(list(map(itemgetter(0), pos[:v_index + 1] + pos[v_index + 2:])))

        if pos[v_index + 2][1] == 'VERB':
            verb = pos[v_index + 2][0]
            if 'Person=3' in pos[v_index][2]:
                verb = get_third_singular_verb(verb)
            return ' '.join(list(map(itemgetter(0), pos[:v_index] + [(verb, '')] + pos[v_index + 3:])))

    if v_index + 1 < len(pos) and pos[v_index + 1][1] == 'VERB':  # 助动词后面跟动词
        verb = (pos[v_index][0] if pos[v_index][0] != 'will' else 'would') + 'n\'t'
        return ' '.join(list(map(itemgetter(0), pos[:v_index] + [(verb, '')] + pos[v_index + 1:])))

    if v_index + 1 < len(pos) and pos[v_index][1] == 'AUX' and pos[v_index + 1] not in ['VERB', 'PART']:
        return ' '.join(list(map(itemgetter(0), pos[:v_index + 1] + [('not', '')] + pos[v_index + 1:])))

    if v_index + 1 < len(pos) and pos[v_index][1] == 'VERB':
        verb = WordNetLemmatizer().lemmatize(pos[v_index][0], 'v')
        feature = f'{pos[v_index][2]}|{pos[v_index - 1][2]}'.replace('|_', '')
        if get_feature(feature, 'Tense', 'Pres') and \
                get_feature(feature, 'Person', '3') and \
                get_feature(feature, 'Number', 'Sing'):
            return ' '.join(
                list(map(itemgetter(0), pos[:v_index] + [('doesn\'t', ''), (verb, '')] + pos[v_index + 1:])))
        elif get_feature(feature, 'Tense', 'Pres'):
            return ' '.join(
                list(map(itemgetter(0), pos[:v_index] + [('don\'t', ''), (verb, '')] + pos[v_index + 1:])))
        else:
            return ' '.join(
                list(map(itemgetter(0), pos[:v_index] + [('didn\'t', ''), (verb, '')] + pos[v_index + 1:])))
    # print(pos)
    return 'None'


def pos_neg_convert(s, stanza_pipline, return_sentiment=False, return_adv_neg=False):
    neg_words = [(' none of ', ' '), (' neither ', ' '), (' no ', ' some '),
                 (' unable ', ' able '), (' few ', ' some '), (' little ', ' some '), (' never ', ' ')]
    doc = stanza_pipline(s)
    if return_sentiment:
        sentiment = sum([s.sentiment for s in doc.sentences]) / len(doc.sentences)
    s = ' ' + s

    poses = [[(word.text, word.upos, word.feats if word.feats else "_", int(word.head)) for word in sent.words] for sent
             in
             doc.sentences]

    if return_adv_neg:
        return [convert(pos, return_adv_neg=True) for pos in poses][0]

    for neg_word in neg_words:
        if neg_word[0] in s:
            if neg_word[0] == ' few ' or neg_word[0] == ' little ':
                if f'a {neg_word[0]}' in s:
                    continue
            s = s.replace(neg_word[0], neg_word[1])
            if neg_word == ' neither ':
                s = s.replace(' nor ', ' and ')
            if return_sentiment:
                return s, sentiment
            return s
    s = s[1:]

    sentence = ' '.join([convert(pos) for pos in poses])

    if return_sentiment:
        return sentence, sentiment
    return sentence

