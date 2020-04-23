import GlobalParament
import utils
from gensim.models import word2vec

#训练模型word2vec
def train(sentences, model_save_path):
    print("开始训练")
    model=word2vec.Word2Vec(sentences=sentences,size=GlobalParament.train_size,window=GlobalParament.train_window)
    model.save(model_save_path)
    print("保存模型结束")


if __name__ == '__main__':
    word='恶寒'
    word1='厥逆'
    word_list=['恶寒','社会']
    type ='wiki'
    #sentences=utils.process_text(GlobalParament.text_alldata,GlobalParament.text_afterprocess_alldata,GlobalParament.stop_words_dir)
    #sentences=utils.load_traintext(GlobalParament.text_afterprocess_partdata_word)
    sentences = utils.load_traintext(GlobalParament.wiki_afterprocess)
    train(sentences,GlobalParament.model_save_path+str(GlobalParament.train_size)+'-'+str(GlobalParament.train_window)+'-'+type+'.model')
    #print (len(sentences))
    #sim_list = []
    model=word2vec.Word2Vec.load(GlobalParament.model_save_path+str(GlobalParament.train_size)+'-'+str(GlobalParament.train_window)+'-'+type+'.model')
    vocab=list(model.wv.vocab.keys())
    #model.wv.save_word2vec_format('embedding.txt')

    #print(model.wv.index2word())  # 获得所有的词汇
    # for word in model.wv.index2word():
    #     print(word, model[word])
    # print(vocab)
    # print(len(vocab))
    with open(GlobalParament.mode_test_path, 'a', encoding=GlobalParament.encoding) as f_writer:
        f_writer.write('\n***************' + type + '*******************\n')
        f_writer.write("句子长度： " + str(len(sentences)) + '\n')
        f_writer.write("词表大小：" + str(len(vocab)) + '\n')
        f_writer.write('window: ' + str(GlobalParament.train_window) + '  size:' + str(GlobalParament.train_size )+ '\n')

        for word in word_list:
            f_writer.write('与 ' + word + ' 比较：\n')
            for e in model.most_similar(positive=word, topn=10):
                f_writer.write(e[0]+'  '+str(e[1])+'\n')
            f_writer.write('------------------------\n')
        #sim_value = model.similarity(word1, word2)
        #f_writer.write(str(sim_value))
        f_writer.write('\n')
        f_writer.flush()
    f_writer.close()







