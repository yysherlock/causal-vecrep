import json
import graphlab as gl

def transfer2gl_format(json_filename, csv_filename, gl_filename):
    """ You can learn how to transfer csv file into gl format"""

    data = json.loads(open(json_filename).read())
    copa_words = []

    with open('copa.txt') as word_file:
         for line in word_file:
                copa_words.append(line.strip())

    with open(csv_filename,'w') as outf:
        outf.write('cause_word effect_word causal_strength\n')
        for i in range(len(data)):
            for j in range(len(data[i])):
                outf.write(copa_words[i]+" "+copa_words[j]+" "+str(data[i][j])+"\n")

#transfer2gl_format('copa_matrix.json', 'copa_matrix.csv')
gl_filename = 'copa_cs'
sf = gl.SFrame.read_csv('copa_matrix.csv')
sf.save(gl_filename)

