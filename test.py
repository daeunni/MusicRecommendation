from Dataset import Dataset
import pre_tag
import word2vec_for_tag

def run():
    data = Dataset()
    #pre_tag.run(data.test,data.n_songs,data.n_tags,data.spr_list,data.tag_tid_id)
    word2vec_for_tag.run(data.total,data.test)


if __name__ == '__main__':
    run()
