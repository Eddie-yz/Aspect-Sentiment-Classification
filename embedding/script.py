import random

aspect_kw_enhace = {'location_n': ['street', 'parking', 'avenue', 'river', 'view'],
                    'location_adj': ['convenient', 'near'],
                    'drinks_n': ['drinks', 'beverage', 'wines', 'margarita', 'sake'],
                    'drinks_adj': ['alcoholic', 'iced', 'bottled'],
                    'food_n': ['food', 'pizza', 'tuna', 'sushi', 'burger'],
                    'food_adj': ['spicy', 'tasty', 'delicious', 'bland', 'savory'],
                    'ambience_n': ['atmosphere', 'room', 'decor', 'music', 'space'],
                    'ambience_adj': ['romantic', 'small', 'cozy', 'tiny'],
                    'service_n': ['tips', 'manager', 'wait', 'waitress', 'servers'],
                    'service_adj': ['rude', 'attentive', 'friendly'],
                    }
# aspect_kw_enhace = {'location_n': ['street', 'avenue', 'river', 'block'],
#                     'location_adj': ['convenient'],
#                     'drinks_n': ['drinks', 'beverage', 'wines', 'margarita', 'sake'],
#                     'drinks_adj': ['bottled'],
#                     'food_n': ['food', 'pizza', 'sushi'],
#                     'food_adj': ['spicy', 'tasty'],
#                     'ambience_n': ['atmosphere', 'room', 'seating'],
#                     'ambience_adj': ['romantic', 'small'],
#                     'service_n': ['tips', 'manager', 'wait', 'waitress', 'servers'],
#                     'service_adj': ['attentive'],
#                     }
cates = ['location', 'drinks', 'food', 'ambience', 'service']

def SelectFromDiffAspects(repeat, word_list):
    res_lst = list()
    for i in range(5):
        target_word_pool = set()
        for _ in range(repeat):
            tmp_lst = list()
            target = random.choice(word_list[i])
            if target in target_word_pool:
                continue
            target_word_pool.add(target)
            tmp_lst.append(target)
            for j in range(5):
                if i == j:
                    continue
                word = random.choice(word_list[j])
                tmp_lst.append(word)
            res_lst.append(tmp_lst)
    return res_lst


if __name__ == '__main__':

    noun_word_lst, adj_word_list = list(), list()
    for cate in cates:
        noun_word_lst.append(aspect_kw_enhace['{}_n'.format(cate)])
    for cate in cates:
        adj_word_list.append(aspect_kw_enhace['{}_adj'.format(cate)])

    res_lst = list()
    res_lst.extend(SelectFromDiffAspects(25, noun_word_lst))
    res_lst.extend(SelectFromDiffAspects(25, adj_word_list))

    with open('betweenAspect.txt', 'w') as f:
        for line in res_lst:
            f.write(' '.join(line)+'\n')