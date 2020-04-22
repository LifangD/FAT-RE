
import json
filename = "dataset/tacred/test.json"
with open (filename) as infile:
    data = json.load (infile)

to_view_list = ['org:number_of_employees/members']
res =  []
count = 0
for d in data:
    if d['relation'] in to_view_list:
        count +=1
        # if count in [11,27,29,30,39,42,43,45,50,52,53,54,55,63,65,66,67,68]:
        if count in [10,13,16,18,19]:
        # if "co-founder" not in d['token']:
            #wds = d['token']
            tokens = d['token']
            sentence = ' '.join (tokens)
            print(sentence)
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
            tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)
            wds = tokens
            head = d['stanford_head']
            deprel = d['stanford_deprel']

            print("```")
            print("graph BT")
            for i, wd in enumerate (wds[:-1]):
                print ("{}_{}-->{}_{}".format (i + 1, wd,  head[i], wds[int (head[i]) - 1]))
            print("```")
            print("---------------------------------------")

        # sentence = ' '.join (tokens)
        # res.append((d['relation'],sentence))

# def by_name(t):
#     return(t[0])
# out = sorted(res,key=by_name)
# for element in out:
#     print(element)