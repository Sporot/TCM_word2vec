import GlobalParament
from jiayan import load_lm
from jiayan import CharHMMTokenizer
import re
import jieba

#去掉回车行
def delete_r_n(line):
    return line.replace('\r','').replace('\n','').strip()

#读取停用词
def get_stop_words(stop_words_dir):
    stop_word=[]

    with open(stop_words_dir,'r',encoding=GlobalParament.encoding) as f:
        for line in f:
            line =delete_r_n(line)
            stop_word.append(line)
    stop_word=set(stop_word)
    return stop_word

#读取特征词
def get_sp_char(word_sp_dir):
    word_sp=[]

    with open(word_sp_dir,'r',encoding=GlobalParament.encoding) as f:
        for line in f:
            line=delete_r_n(line)
            word_sp.append(line)
    word_sp=set(word_sp)
    return word_sp

#读取分词列表
def get_tag_word(word_dir):
    word_tag=[]

    with open(word_dir,'r',encoding=GlobalParament.encoding) as f:
        for line in f:
            line=delete_r_n(line)
            word_list=line.split(' ')
            #print(word_list)
            for word in word_list:
                word_tag.append(word)
    word_tag=set(word_tag)
    return word_tag


#按空格分字
def space_cut(content,stop_word):
    char_list=[]

    if content is not None and content !='':
        for char in content:
            if char not in stop_word and  '\u4e00' <= char <= '\u9fa5':
                char_list.append(char)
    return char_list

#利用jiayan分词
def jiayan_cut(content,stop_word_dir,load_lm_dir):
    stop_word=get_stop_words(stop_word_dir)
    lm=load_lm(load_lm_dir)
    tokenizer=CharHMMTokenizer(lm)
    word_list=[]

    if content !='' and content is not None:
        seg_list=tokenizer.tokenize(content)
        for word in seg_list:
            if word not in stop_word and  '\u4e00' <= word <= '\u9fa5':
                word_list.append(word)

    return word_list

def jiayan_cut_nostop(content, load_lm_dir):
    lm = load_lm(load_lm_dir)
    tokenizer = CharHMMTokenizer(lm)
    word_list=[]
    if content != '' and content is not None:
        seg_list = tokenizer.tokenize(content)
        for word in seg_list:
            word_list.append(word)

    return " ".join(word_list)

#利用jiayan对样本词分词
def jiayan_cut_sample(content,load_lm_dir):
    #stop_word=get_stop_words(stop_word_dir)
    lm=load_lm(load_lm_dir)
    tokenizer=CharHMMTokenizer(lm)
    word_list=[]

    if content !='' and content is not None:
        seg_list=tokenizer.tokenize(content)
        for word in seg_list:
            #if word not in stop_word and  '\u4e00' <= word <= '\u9fa5':
                word_list.append(word)

    return word_list

#清除不在词汇表中的字
def clear_char_from_vocab(char_list, vocab):
    new_char_list =[]

    for char in char_list:
        if char in vocab:
            new_char_list.append(char)

    return new_char_list


#处理文本
def process_text(text_dir, after_process_dir, stop_words_dir):
    stop_word=get_stop_words(stop_words_dir)
    sentences =[]
    count = 0;

    f_writer=open(after_process_dir,'w',encoding=GlobalParament.encoding)
    with open(text_dir,'r',encoding=GlobalParament.encoding) as f_reader:
        for line in f_reader:
            line=delete_r_n(line)
            char_list=space_cut(line,stop_word)
            if len(char_list)>0:
                sentences.append(char_list)
                f_writer.write(" ".join(char_list)+'\n')
                f_writer.flush()
            count=count+1
            print(count)
    f_writer.close()
    return sentences

#加载处理后的文本数据
def load_traintext(text_afterprocess_alldata):
    sentences=[]

    with open (text_afterprocess_alldata,'r',encoding=GlobalParament.encoding) as f:
        for line in f:
            line=delete_r_n(line)
            char_list=line.split(' ')
            sentences.append(char_list)
    return sentences



#利用jiayan对文本进行分词
def process_text_jiayan(text_dir,after_process_dir,stop_words_dir,load_lm_dir):
    sentences =[]
    count=0

    f_writer=open(after_process_dir,'w',encoding=GlobalParament.encoding)
    with open(text_dir,'r',encoding=GlobalParament.encoding) as f_reader:
        for line in f_reader:
            line=delete_r_n(line)
            word_list=jiayan_cut(line,stop_words_dir,load_lm_dir)
            if len(word_list) > 0:
                sentences.append(word_list)
                f_writer.write(" ".join(word_list) + '\n')
                f_writer.flush()
                count = count + 1
                print(count)
    f_writer.close()
    return sentences


#利用jiayan对样本集进行分词
def process_text_jiayan_sample(text_dir,after_process_dir,load_lm_dir):
    sentences =[]
    count=0

    f_writer=open(after_process_dir,'w',encoding=GlobalParament.encoding)
    with open(text_dir,'r',encoding=GlobalParament.encoding) as f_reader:
        for line in f_reader:
            line=delete_r_n(line)
            word_list=jiayan_cut_sample(line,load_lm_dir)
            if len(word_list) > 0:
                sentences.append(word_list)
                f_writer.write(" ".join(word_list) + '\n')
                f_writer.flush()
                count = count + 1
                print(count)
    f_writer.close()
    return sentences

#将样本集合成一个文档
def sample_concat(sample_dir,sample_concat_dir):
    sentences=[]
    f_writer=open(sample_concat_dir,'w',encoding=GlobalParament.encoding)

    with open(sample_dir,'r',encoding=GlobalParament.encoding) as f_reader:
        for line in f_reader:
            #line=delete_r_n(line)
            if len(line)>0:
                char=line[0]
                print(type(char))
                sentences.append(char)
                f_writer.write(char)
    return sentences

def dictionary_contact(dictionary_dir,dictionary_concat_dir):
    sentences = []
    count=0
    f_writer = open(dictionary_concat_dir, 'w', encoding=GlobalParament.encoding)

    with open(dictionary_dir, 'r', encoding=GlobalParament.encoding) as f_reader:
        for line in f_reader:
            count=count+1
            line=delete_r_n(line)
            if len(line) > 0:
                if count==20:
                    f_writer.write(line+'\n')
                    count=0
                sentences.append(line)
                f_writer.write(line+',')
    return sentences

def book_tag(book_tag_dir,book_word_dir):
    f_writer=open(book_word_dir,'w',encoding=GlobalParament.encoding)
    with open(book_tag_dir,'r',encoding=GlobalParament.encoding) as f_reader:
        for line in f_reader:
            seg = re.sub("[0-9\.]", "", line)
            line_list=delete_r_n(seg)
            line_list=line_list.split(' ')
            for word in line_list:
                print(len(word))
                f_writer.write(word+'\n')
                f_writer.flush()
            f_writer.write('\n')
    f_writer.close()
    return

if __name__ == '__main__':
    # stop_word=get_stop_words(GlobalParament.stop_words_dir)
    # content='太阳病，发热无汗'
    # print(space_cut(content,stop_word))
    # sentences1=process_text_jiayan(GlobalParament.test_load_dir,GlobalParament.test_load_afterprocess_dir,GlobalParament.stop_words_dir,GlobalParament.load_lm_path)
    # print(sentences1)
    # sentences2=load_traintext(GlobalParament.test_load_afterprocess_dir)
    # print(sentences2)
    #
    # if sentences2==sentences1:
    #     print("the same")
    # else:
    #     print("different")
    #process_text(GlobalParament.dictionary_path,GlobalParament.dictionary_afterprocess_path,GlobalParament.stop_words_dir)
    #process_text(GlobalParament.sample_path,GlobalParament.sample_afterprocess_path,GlobalParament.stop_words_dir)
    content='妇人脏躁，喜悲伤欲哭，象如神灵所作，数欠伸，甘麦大枣汤主之。'
    #print(jiayan_cut(content,GlobalParament.stop_words_dir,GlobalParament.load_lm_path))
    print(jiayan_cut_nostop(content,GlobalParament.load_lm_path))
    seg_list=jieba.cut(content)  #
    print("Full Mode: " + ",".join(seg_list) ) # 全模式
    #process_text_jiayan(GlobalParament.text_alldata,GlobalParament.text_afterprocess_alldata_word,GlobalParament.stop_words_dir,GlobalParament.load_lm_path)
    #process_text_jiayan(GlobalParament.sample_concat_dir,GlobalParament.sample_afterprocess,GlobalParament.stop_words_dir,GlobalParament.load_lm_path)
    #dictionary_contact(GlobalParament.dictionary_dir,GlobalParament.dictionary_concat_dir)
    #process_text_jiayan_sample(GlobalParament.sample_concat_dir,GlobalParament.sample_afterprocess_all,GlobalParament.load_lm_path)
    # process_text_jiayan_sample(GlobalParament.book1_dir,GlobalParament.book1_tag_dir,GlobalParament.load_lm_path)
    # process_text_jiayan_sample(GlobalParament.book2_dir,GlobalParament.book2_tag_dir,GlobalParament.load_lm_path)
    # process_text_jiayan_sample(GlobalParament.book3_dir,GlobalParament.book3_tag_dir,GlobalParament.load_lm_path)
    # book_tag(GlobalParament.book1_tag_dir,GlobalParament.book1_word_dir)
    # book_tag(GlobalParament.book2_tag_dir,GlobalParament.book2_word_dir)
    # book_tag(GlobalParament.book3_tag_dir,GlobalParament.book3_word_dir)
    # word_tag=get_tag_word(GlobalParament.word_tag_dir)
    # print(word_tag)



