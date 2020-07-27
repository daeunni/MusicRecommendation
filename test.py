from Dataset import Dataset
import pre_tag

def run():
    data = Dataset()
    pre_tag.run(data.test,data.n_songs,data.n_tags,data.spr_list,data.tag_tid_id)

if __name__ == '__main__':
    run()
