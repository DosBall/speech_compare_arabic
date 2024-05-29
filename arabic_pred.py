from keras.models import load_model
import numpy as np
import librosa
from librosa import feature
import argparse
import os, time
from flask import Flask
from flask import request

target_sr = 22050
top_db1 = 20


def make_predictions0(model, file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfccs = feature.mfcc(y=audio, sr=target_sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    features = mfccs_scaled.reshape(1, mfccs_scaled.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return predicted_class_index, predicted_vector


def make_predictions3(model, file_path1, n):
    y1, sr1 = librosa.load(file_path1, sr=target_sr)
    y_t1, sr_t1 = librosa.effects.trim(y1, top_db=top_db1)
    mfccs = feature.mfcc(y=y_t1, sr=target_sr, n_mfcc=n)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    mfccs_scaled1 = np.std(mfccs, axis=1)
    ans = np.concatenate((mfccs_scaled, mfccs_scaled1), axis=0)
    features = ans.reshape(1, ans.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index1 = np.argmax(predicted_vector, axis=-1)
    return predicted_class_index1, predicted_vector


def make_predictions1(model, file_path, n=40):
    y1, sr1 = librosa.load(file_path, sr=target_sr)
    y_t1, sr_t1 = librosa.effects.trim(y1, top_db=top_db1)
    mel = feature.melspectrogram(y=y_t1, sr=target_sr, n_mels=n)
    mel_scaled = np.mean(mel.T, axis=0)
    mel_scaled1 = np.std(mel.T, axis=0)
    ans = np.concatenate((mel_scaled, mel_scaled1), axis=0)
    features = ans.reshape(1, ans.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return predicted_class_index, predicted_vector

def make_predictions2(model, file_path, n1, n2):
    y1, sr1 = librosa.load(file_path, sr=target_sr)
    y_t1, sr_t1 = librosa.effects.trim(y1, top_db=top_db1)
    mfccs = feature.mfcc(y=y1, sr=target_sr, n_mfcc=n1)
    mfccs_scaled = np.mean(mfccs, axis=1)
    mfccs_scaled1 = np.std(mfccs, axis=1)
    mel = feature.melspectrogram(y=y_t1, sr=target_sr, n_mels=n2)
    mel_scaled = np.mean(mel, axis=1)
    mel_scaled1 = np.std(mel, axis=1)
    ans = np.concatenate((mfccs_scaled, mfccs_scaled1, mel_scaled, mel_scaled1), axis=0)
    features = ans.reshape(1, ans.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return predicted_class_index, predicted_vector

def make_predictions4(model, file_path1, n):
    y1, sr1 = librosa.load(file_path1, sr=target_sr)
    y_t1, sr_t1 = librosa.effects.trim(y1, top_db=top_db1)
    mfccs = feature.mfcc(y=y_t1, sr=target_sr, n_mfcc=n)
    ans = np.mean(mfccs.T, axis=0)
    #mfccs_scaled1 = np.std(mfccs, axis=1)
    #ans = np.concatenate((mfccs_scaled, mfccs_scaled1), axis=0)
    features = ans.reshape(1, ans.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index1 = np.argmax(predicted_vector, axis=-1)
    return predicted_class_index1, predicted_vector

def make_predictions5(model, file_path1, n):
    y1, sr1 = librosa.load(file_path1, sr=target_sr)
    #y_t1, sr_t1 = librosa.effects.trim(y1, top_db=top_db1)
    mfccs = feature.mfcc(y=y1, sr=target_sr, n_mfcc=n)
    ans = np.mean(mfccs.T, axis=0)
    features = ans.reshape(1, ans.shape[0], 1)
    predicted_vector = model.predict(features)
    predicted_class_index1 = np.argmax(predicted_vector, axis=-1)
    return predicted_class_index1, predicted_vector


# Load the trained model
#trained_model = load_model('best_model1.h5', compile=True)
#trained_model1 = load_model('mel_spec_trim_model1.h5', compile=True)
#trained_model2 = load_model('mel256.h5', compile=True)
#trained_model3 = load_model('mel_mfcc_296_trim.h5', compile=True)
#trained_model4 = load_model('mfcc40.h5', compile=True)
#trained_model5 = load_model('mfcc_200_std_trim.h5', compile=True)
#trained_model6 = load_model('mfcc_100_trim.h5', compile=True)
trained_model7 = load_model('mfcc_100_trim_small.h5', compile=True)
trained_model8 = load_model('mfcc_40_trim_small.h5', compile=True)
#trained_model9 = load_model('mel_256_trim_small.h5', compile=True)
trained_model10 = load_model('mfcc_100_trim_small_100.h5', compile=True)
#trained_model11 = load_model('mfcc_100_small_100.h5', compile=True)
#trained_model12 = load_model('mfcc_50+50_small_100.h5', compile=True)
#trained_model13 = load_model('mfcc_50+50_small_200.h5', compile=True)
#trained_model14 = load_model('mfcc_20+20_small_100.h5', compile=True)
#trained_model15 = load_model('mfcc_20+20_small_40.h5', compile=True)
#trained_model16 = load_model('mfcc_100_trim_small_20.h5', compile=True)
#trained_model17 = load_model('mfcc_20+20_small_30.h5', compile=True)

d_arab = {0: "الف", 1: "باء", 2: "تاء", 3: "ثاء", 4: "جيم", 5: "حاء", 6: "خاء",
          7: "دال", 8: "ذال", 9: "راء", 10: "زاي", 11: "سين", 12: "شين", 13: "صاد",
          14: "ضاد", 15: "طاء", 16: "ظاء", 17: "عين", 18: "غين", 19: "فاء", 20: "قاف",
          21: "كاف", 22: "لام", 23: "ميم", 24: "نون", 26: "هاء", 25: "واو", 27: "ياء"}
d_ru = {0: "алиф", 1: "ба", 2: "та", 3: "tha", 4: "джим", 5: "ха", 6: "кхъа",
        7: "даль", 8: "thаль", 9: "ра", 10: "за", 11: "син", 12: "щин", 13: "съад",
        14: "дъад", 15: "тъа", 16: "thъа", 17: "'аин", 18: "гъаин", 19: "фа", 20: "къаф",
        21: "кяф", 22: "лям", 23: "мим", 24: "нун", 26: "гха", 25: "уау", 27: "йа"}
# Use the trained model for predictions

my_path1 = "arab"
my_path2 = "audios_zhaina/cor"
my_path3 = "audios_zhoha/cor"
my_path4 = "audios_zhoha/incor"
my_path5 = "askar"


def test1dir(my_path):
    files = os.listdir(my_path)
    print(files)
    mas1, mas2, mas3 = [], [], []
    for i in files:
        try:
            #pred, vec = make_predictions3(trained_model5, my_path + "/" + i, 100)
            #pred, vec = make_predictions4(trained_model6, my_path + "/" + i, 100)
            pred, vec = make_predictions4(trained_model7, my_path + "/" + i, 100)
            #pred, vec = make_predictions3(trained_model8, my_path + "/" + i, 20)
            #pred, vec = make_predictions1(trained_model9, my_path + "/" + i, 128)
            #pred, vec = make_predictions4(trained_model10, my_path + "/" + i, 100)
            #pred, vec = make_predictions5(trained_model11, my_path + "/" + i, 100)
            #pred, vec = make_predictions3(trained_model12, my_path + "/" + i, 50)
            #pred, vec = make_predictions3(trained_model13, my_path + "/" + i, 50)
            #pred, vec = make_predictions3(trained_model14, my_path + "/" + i, 20)
            #pred, vec = make_predictions3(trained_model15, my_path + "/" + i, 20)
            #pred, vec = make_predictions4(trained_model16, my_path + "/" + i, 100)
            #pred, vec = make_predictions3(trained_model17, my_path + "/" + i, 20)

            pred, vec, vec1 = int(pred[0]), vec[0], np.sort(np.copy(vec))[0][::-1]
            print("mffc  real:", i, "pred:", pred + 1, d_arab[pred], d_ru[pred])
            print(vec)
            print(vec1)
            temp1 = str(vec[int(i[:3]) - 1])
            j = 1
            while temp1 != str(vec1[j - 1]):
                j += 1
            mas1.append(j)
            temp_f = round(float(vec1[0]) / float(vec1[j - 1]), 3)
            mas2.append(temp_f)
            mas3.append(round(vec[int(i[:3]) - 1] * 100, 1))
        except Exception as e:
            print("fail ", i)
            print(e)
        time.sleep(1)

    print(my_path)
    print(*[mas1, mas2, sum(mas1) / len(mas1), sum(mas2) / len(mas2), mas3], sep="\n")


def test1(path1, make1, model1, n1, a1):
    pred1, vec1 = make1(model1, path1, n1)
    pred1, vec1, vec11 = int(pred1[0]), vec1[0], np.sort(np.copy(vec1))[0][::-1]
    temp1, j = str(vec1[a1]), 1
    while temp1 != str(vec11[j - 1]):
        j += 1
    return j, round(vec1[a1] * 100, 1)


def test_models(path1, a=None, k=5):
    if a is None:
        temp = path1.rfind("/")
        if not temp:
            a = int(path1[:3]) - 1
        else:
            a = int(path1[temp + 1:temp + 4]) - 1
    j1, p1 = test1(path1, make_predictions4, trained_model10, 100, a)
    j2, p2 = test1(path1, make_predictions3, trained_model8, 20, a)
    j3, p3 = test1(path1, make_predictions4, trained_model7, 100, a)
    print("j:", j1, j2, j3, "p:", p1, p2, p3)
    return 1 if j1 <= k or j2 <= k or j3 <= k else 0  # 3 or
    # jj = sorted([j1, j2, j3])
    #return 1 if jj[0] <= k or jj[1] <= k * 2 else 0  # 1 or 2
    #return 1 if jj[0] <= k and jj[2] <= k*3 else 0  # 1 and 3


def test_dir(path_dir):
    files = os.listdir(path_dir)
    #print(files)
    d0 = []
    for i in files:
        a = int(i[:3]) - 1
        #d0[a] = "YES" if test_models(path_dir + "/" + i, a=a) else "NO"
        if test_models(path_dir + "/" + i, a=a) == 0:
            d0.append((a, d_ru[a]))
    return path_dir, d0


def test_dirs(*dirs):
    ans = []
    for i in dirs:
        ans.append(test_dir(i))
    print(*ans, sep="\n")


#print(test1("arab/001-alif.mp3", make_predictions4, trained_model10, 100, 0))
#print(test_models("arab/001-alif.mp3", 0))
#print(test_dir(my_path1))
#test_dirs(my_path1, my_path2, my_path3, my_path4)
#test1dir(my_path1)


"""
test1dir(my_path1)
test1dir(my_path2)
test1dir(my_path3)
test1dir(my_path4)
test1dir(my_path5)
"""

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs=1, help="files to compair")
    args = parser.parse_args()
    predicted_class_index = make_predictions1(trained_model, args.file[0])
    print("Predicted Label:", predicted_class_index)
"""


def test_models1(path1, a=None, k=None):
    if k is None:
        k = 4
    if a is None:
        temp = path1.rfind("/")
        if not temp:
            a = int(path1[:3]) - 1
        else:
            a = int(path1[temp + 1:temp + 4]) - 1
    j1, p1 = test1(path1, make_predictions4, trained_model10, 100, a)
    j2, p2 = test1(path1, make_predictions3, trained_model8, 20, a)
    j3, p3 = test1(path1, make_predictions4, trained_model7, 100, a)
    ans = 1 if j1 <= k or j2 <= k or j3 <= k else 0
    return ans, [j1, j2, j3], [p1, p2, p3]

app = Flask('__name__')


@app.route('/answer', methods=['GET', 'POST'])
def answer():

    speach_path = request.args.get('speach')
    letter_num = request.args.get('letter')
    k = request.args.get('k')
    if k is not None:
        k = int(k)
    if letter_num is not None:
        letter_num = int(letter_num)
    #print(speach_path, letter_num, k)
    ans, j, p = test_models1(speach_path, letter_num, k)
    return str(ans) + "\nМеста: " + str(j) + "\nПроценты: " + str(p)
    #return str(ans)


app.run(host='0.0.0.0', port=80)


