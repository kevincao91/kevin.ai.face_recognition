def copy_info(scr_path, tar_path):

    with open(scr_path, 'r') as f:
        lines = f.readlines()

    num_line = lines[0]
    pos_num = int(num_line)
    pos_pairs_lines = lines[1:pos_num+1]
    print(pos_pairs_lines[0], pos_pairs_lines[-1])
    neg_pairs_lines = lines[pos_num+1:]
    print(neg_pairs_lines[0], neg_pairs_lines[-1])

    with open(tar_path, 'w') as f:
        f.writelines(num_line)
        for i in range(pos_num):
            neg_pairs_words = neg_pairs_lines[i].split()
            comb_line = pos_pairs_lines[i][:-1] + '\t' + neg_pairs_words[0] + '\t' + neg_pairs_words[1] + '\n'
            f.writelines(comb_line)


def create():
    pairsDevTrain_path = '/home/kevin/文档/LFW/pairsDevTrain.txt'
    pairsDevTest_path = '/home/kevin/文档/LFW/pairsDevTest.txt'
    tripletDevTrain_path = '/home/kevin/文档/LFW/tripletDevTrain.txt'
    tripletDevtest_path = '/home/kevin/文档/LFW/tripletDevTest.txt'

    copy_info(pairsDevTrain_path, tripletDevTrain_path)

    copy_info(pairsDevTest_path, tripletDevtest_path)


if __name__ == '__main__':
    create()
