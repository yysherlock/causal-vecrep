"""
Simple Regression using 
similarity and causality as raw features

"""
import numpy as np
import graphlab as gl

copa_word_set = set({})

def generate_copacs(csfn, copacsfn,copawordfn):
    
    with open(csfn) as csf, open(copawordfn) as copawordf, open(copacsfn,'w') as copacsf:
        for line in copawordf: copa_word_set.add(line.strip())
        for line in csf:
            cause,effect,freq = line.strip().split('\t')
            if cause in copa_word_set and effect in copa_word_set:
                copacsf.write(line)

def get_copacs(copa_matrix_fn, csv_fn):
    with open(copa_matrix_fn) as f, open(csv_fn, 'w') as outf:
        outf.write(','.join(f.readline().strip().split()))
        outf.write('\n')
        for line in f:
            cause,effect,score = line.strip().split()
            outf.write(','.join([cause,effect,score]))
            outf.write('\n')

def save2disc(copa_csv,sffn):
    sf = gl.SFrame.read_csv(copa_csv, delimiter=",")    
    sf.save(sffn)

def get_word_dict(word,role):
    cols = sf.column_names()
    cols.remove(role+'_word')
    roledword = sf[sf[role+'_word']==word][cols]
    roleddict = dict(roledword.to_numpy())
    return roleddict

def dict_add(dict1,dict2):
    for k,v in dict2.items():
        dict1.setdefault(k,0.0)
        dict1[k] += float(v)
    return dict1

def get_cosineDist_feature(cause_sen, effect_sen):
    cause_sen_dict, effect_sen_dict = {}, {}

    for cause in cause_sen:
        cause_sen_dict = dict_add(cause_sen_dict,get_word_dict(cause,"cause"))
    
    for effect in effect_sen:
        effect_sen_dict = dict_add(effect_sen_dict,get_word_dict(effect,"effect"))

    return gl.distances.cosine(cause_sen_dict, effect_sen_dict)

def get_causality_feature(cause_sen, effect_sen):
    z = len(cause_sen) + len(effect_sen)
    s = 0.0 
    for cause in cause_sen:
        for effect in effect_sen:
            try:
                print '--'+cause+'--'+effect+'--'
                print sf[(sf['cause_word']==cause) & (sf['effect_word']==effect)]['causal_strength'][0]

                score = float(sf[(sf['cause_word']==cause) & (sf['effect_word']==effect)]['causal_strength'][0])
                s += score
            except Exception, e:
                print e
                pass
    
    return  s / z

def extract_features(cause_sen1, effect_sen1, cause_sen2, effect_sen2):
    f1 = get_cosineDist_feature(cause_sen1, effect_sen1)
    f2 = get_causality_feature(cause_sen1, effect_sen1)
    f3 = get_cosineDist_feature(cause_sen2, effect_sen2)
    f4 = get_causality_feature(cause_sen2, effect_sen2)
    print f3-f1,f4-f2
    return (f3-f1, f4-f2)

# filename is the log
# cause_sen1:
# effect_sen1:
# cause_sen2:
# effect_sen2:
# s1:
# s2:
# id: true / false

def generate_features(filename, featfn):
    with open(filename) as f, open(featfn,'w') as outf:
        f.readline()
        outf.write('cosine_distance_diff,causality_diff,label\n')
        cnt = 0
        cause_sen1, effect_sen1, cause_sen2, effect_sen2 = [],[],[],[]
        s1,s2 = 0.0,0.0
        idx, result = "",  ""
        label = 0
        for line in f:
            line = line.strip('\n')
            if line.endswith(', '): line = line.strip(', ')
            if cnt > 0 and cnt % 7 == 0:
                # generate features
                print cause_sen1, effect_sen1, cause_sen2, effect_sen2
                feat1,feat2 = extract_features(cause_sen1, effect_sen1, cause_sen2, effect_sen2)
                if result=="true":
                    if s1 > s2: label = 1
                    else: label = -1
                elif result=="false":
                    if s1 > s2: label = -1
                    else: label = 1
                else: print 'error 0'
                outf.write(str(feat1)+','+str(feat2)+','+str(label)+'\n')
                outf.flush()

                # reset
                cause_sen1, effect_sen1, cause_sen2, effect_sen2 = [],[],[],[]
                s1,s2,label = .0, .0, 0
                idx, result = "",""
            if cnt % 7 == 6:
                idx, result = line.split(':')
            else:
                head,content = line.split(': ')
                print content+'--'

                # switch in python
                exec({"cause_sen1": "cause_sen1 = content.split(', ')",
                "effect_sen1": "effect_sen1 = content.split(', ')",
                "cause_sen2": "cause_sen2 = content.split(', ')",
                "effect_sen2": "effect_sen2 = content.split(', ')",
                "s1": "s1 = float(content)",
                "s2": "s2 = float(content)",
                }.get(head, "print 'error'"))
            
            cnt += 1
            print cnt

def get_numpy_data(data_frame, features, output):
    #data_frame['constant'] = 1
    #features = ['constant'] + features
    features_sframe = data_frame[features]
    features_matrix = features_sframe.to_numpy()
    output_sarray = data_frame[output]
    output_array = output_sarray.to_numpy()
    return (features_matrix, output_array)

def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions

def delta_objective_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance, maxiter = 3000):
    
    converged = False
    iterations = 0
    weights = np.array(initial_weights)
    while iterations < maxiter and not converged:
        predictions = predict_ouput(feature_matrix, weights)
        print 'iteration',str(iterations),', cost: ', np.dot(output, predictions), 'the smaller the better'

        derivative = np.dot(output, feature_matrix)
        weights -= step_size * derivative
        gradient_magnitude = sqrt(np.sum(derivative**2))
        
        if gradient_magnitude < tolerance:
            converged = True

    return weights


sf = gl.SFrame('copa_cs/')

if __name__=="__main__":
    #get_copacs('copa_matrix.csv','copa_cs.csv')
    #save2disc('copa_cs.csv','copa_cs')
    generate_features('lambda=0.9_log.txt', 'simple_data.csv')
    save2disc('simple_data.csv','simple_data')
    data_frame = gl.SFrame('simple_data')
    
    initial_weights = np.array([1.,1.])
    step_size = 1e-7
    tolearance = 2.5e2
    simple_features = ['cosine_distance_diff','causality_diff']
    my_output = 'label'
    (simple_feature_matrix, output) = get_numpy_data(data_frame, simple_features, output, initial_weights, step_size, tolearance)
    simple_weights = delta_objective_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
    print 'weights:', simple_weights

