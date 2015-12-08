import json
import graphlab as gl
import numpy as np

def copa_matrix_transfer2gl_format(json_filename, csv_filename, gl_filename):
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
    sf = gl.SFrame.read_csv(csv_filename)
    sf.save(gl_filename)

#copa_matrix_transfer2gl_format('copa_matrix.json', 'copa_matrix.csv', 'copa_cs')

def copa_vector_transfer2gl_format(csv_filename, gl_filename,col_json_filename=None, row_json_filename=None, col_role="cause", row_role="effect"):
    if not col_json_filename and  not row_json_filename:
        print "Both col_file and row_file are None."
        exit(1)

    col_data, row_data = None, None

    try:
        col_data = json.loads(open(col_json_filename).read())
        row_data = json.loads(open(row_json_filename).read())
    except : pass

    copa_words = []

    with open('copa.txt') as word_file:
        for line in word_file:
            copa_words.append(line.strip())

    outf = open(csv_filename,'w')
    
    if col_data:
        # col vector head
        for i in range(len(col_data[0])):
            outf.write(col_role+"_"+copa_words[i]+" ")
    if row_data:
        row_data = np.array(row_data).T
        # row vector head
        for i in range(len(col_data[0])):
            outf.write(row_role+"_"+copa_words[i])
            if i!=len(col_data[0])-1: outf.write(" ")

        outf.write('\n')

    for i in range(len(col_data)):
        for j in range(len(col_data[i])):
            outf.write(str(col_data[i][j])+" ")
        for j in range(len(row_data[i])):
            outf.write(str(row_data[i][j]))
            if j!=len(row_data[i])-1: outf.write(" ")
        outf.write('\n')
    
    outf.close()

    sf = gl.SFrame.read_csv(csv_filename, delimiter=" ")
    sf.save(gl_filename)

#copa_vector_transfer2gl_format('svd_vec.csv', 'svd_vec', 'u.json', 'v.json', "cause", "effect")


sf = gl.SFrame('svd_vec/')
def calculate_prod_distance(outfilename):
    outf = open(outfilename,'w')
    names = sf.column_names()
    for cause in names:
        if not cause.startswith('cause_'): continue
        for effect in names:
            if not effect.startswith('effect_'): continue
            cs = (sf[cause]*sf[effect]).sum()
            outf.write(cause.split('_')[1]+'\t'+effect.split('_')[1]+'\t'+str(cs)+'\n')
    outf.close()

calculate_prod_distance('vector_cs.txt')
