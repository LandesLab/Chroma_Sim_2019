import sys
from column_classes import *
import datetime

def c_runner(i, j, k, l, directory, mol_limit, con_ads_len):
    name = 'raretime_'+str(i)+'_len_'+str(j)+'adProb_'+str(k)+'prev_'+str(l)
    print(name + ' ' + str(datetime.datetime.now()))
    stat_phase_dict = {'res': (int(j),5), 'contPDFs': True}
    col_dict = ChromaColumn(name = name, make_paths = False, stat_dict = stat_phase_dict, molecule_limit = mol_limit, cons_ads = k, directory = directory, competitive = False)
    col_dict.newSiteType(name = 'short', distrib = rnd.exponential, d_options = {'scale': con_ads_len}, prevalence = 1-l)
    col_dict.newSiteType(name = 'long', distrib = rnd.exponential, d_options = {'scale': i}, prevalence = l)
    col_dict.performElution()
    col_dict.gatherData()
    #col_dict.dumpParameterstoJSON()
    print(col_dict.sayCurrentRunDirectory())
    return 0

def Nicks_c_runner(i, mol_limit, directory):
    # Here, the code assumes that all the data for the site types comes from  the tabel below and that the column has some standard values. Further, it assumes an index value has been input to allow referencing to the tables below.
    names = ['0mMNaCL','1mMNaCL','10mMNaCL','50mMNaCL','100mMNaCL','500mMNaCL','1000mMNaCL']
    short_A1s = np.array([0.39594, 0.37238, 0.51115, 0.65381, 0.6984, 0.69573, 0.53834])
    short_t1s = np.array([109.845, 52.99631, 78.61804, 51.16535, 61.42786, 53.78534, 56.721])
    long_A2s = np.array([0.55363, 0.55569, 0.39378, 0.37198, 0.27522, 0.33825, 0.45603])
    long_t2s = np.array([287.518, 254.80877, 270.99632, 189.4863, 249.9056, 211.28774, 201.23429])
    name = 'NicksData_arrayVal_'+str(i)
    print(name + ' ' + str(datetime.datetime.now()))
    stat_phase_dict = {'res': (int(100),5), 'contPDFs': True}
    col_dict = ChromaColumn(name = name, make_paths = False, stat_dict = stat_phase_dict, molecule_limit = mol_limit, cons_ads = 0.5, directory = directory, competitive = False)
    col_dict.newSiteType(name = 'short', distrib = rnd.exponential, d_options = {'scale': short_t1s[i]}, prevalence = short_A1s[i])
    col_dict.newSiteType(name = 'long', distrib = rnd.exponential, d_options = {'scale': long_t2s[i]}, prevalence = long_A2s[i])
    col_dict.performElution()
    col_dict.gatherData()
    print(col_dict.sayCurrentRunDirectory())
    return 0

def Nicks_c_runner_2(i, mol_limit, directory):
    names = ['0mMNaCL','1mMNaCL','10mMNaCL','50mMNaCL','100mMNaCL','500mMNaCL','1000mMNaCL']
    short_t1s = np.array([180.842,  107.140,    123.968,    67.647, 71.6878,    68.1618,    70.658])
    long_t2s = np.array([1373,      1272,       1057,       826,    859,        863,        899])
    col_len = 100
    cons_ads = 0.5
    name = 'NicksData_arrayVal_'+str(i)+'_colLen'+str(col_len)+'_consAd'+str(cons_ads)
    print(name + ' ' + str(datetime.datetime.now()))
    stat_phase_dict = {'res': (int(col_len),5), 'contPDFs': True}
    col_dict = ChromaColumn(name = name, make_paths = False, stat_dict = stat_phase_dict, molecule_limit = mol_limit, cons_ads = cons_ads, directory = directory, competitive = False)
    col_dict.newSiteType(name = 'short', distrib = rnd.exponential, d_options = {'scale': short_t1s[i]}, prevalence = 0.99)
    col_dict.newSiteType(name = 'long', distrib = rnd.exponential, d_options = {'scale': long_t2s[i]}, prevalence = 0.01)
    col_dict.performElution()
    col_dict.gatherData()
    print(col_dict.sayCurrentRunDirectory())
    return 0

def Nicks_c_runner_3(i, j, k, l, mol_limit, directory):
    names = ['0mMNaCL','1mMNaCL','10mMNaCL','50mMNaCL','100mMNaCL','500mMNaCL','1000mMNaCL']
    short_t1s = np.array([180.842,  107.140,    123.968,    67.647, 71.6878,    68.1618,    70.658])
    long_t2s = np.array([1373,      1272,       1057,       826,    859,        863,        899])
    name = 'NicksData_arrayVal_'+str(i)+'_len_'+str(j)+'adProb_'+str(k)+'prev_'+str(l)
    print(name + ' ' + str(datetime.datetime.now()))
    stat_phase_dict = {'res': (int(j),5), 'contPDFs': True}
    col_dict = ChromaColumn(name = name, make_paths = False, stat_dict = stat_phase_dict, molecule_limit = mol_limit, cons_ads = float(k), directory = directory, competitive = False)
    col_dict.newSiteType(name = 'short', distrib = rnd.exponential, d_options = {'scale': short_t1s[i]}, prevalence = 1-float(l))
    col_dict.newSiteType(name = 'long', distrib = rnd.exponential, d_options = {'scale': long_t2s[i]}, prevalence = float(l))
    col_dict.performElution()
    col_dict.gatherData()
    print(col_dict.sayCurrentRunDirectory())
    return 0

def Nicks_c_runner_4(i, j, k, l, mol_limit, directory):
    names = ['0mMNaCL','1mMNaCL','10mMNaCL','50mMNaCL','100mMNaCL','500mMNaCL','1000mMNaCL']
    short_A1s = np.array([0.39594, 0.37238, 0.51115, 0.65381, 0.6984, 0.69573, 0.53834])
    short_t1s = np.array([109.845, 52.99631, 78.61804, 51.16535, 61.42786, 53.78534, 56.721])
    long_A2s = np.array([0.55363, 0.55569, 0.39378, 0.37198, 0.27522, 0.33825, 0.45603])
    long_t2s = np.array([287.518, 254.80877, 270.99632, 189.4863, 249.9056, 211.28774, 201.23429])
    name = 'NicksData_arrayVal_'+str(i)+'_len_'+str(j)+'adProb_'+str(k)+'prev_'+str(l)
    print(name + ' ' + str(datetime.datetime.now()))
    stat_phase_dict = {'res': (int(j),5), 'contPDFs': True}
    col_dict = ChromaColumn(name = name, make_paths = False, stat_dict = stat_phase_dict, molecule_limit = mol_limit, cons_ads = float(k), directory = directory, competitive = False)
    col_dict.newSiteType(name = 'short', distrib = rnd.exponential, d_options = {'scale': short_t1s[i]}, prevalence = short_A1s[i])
    col_dict.newSiteType(name = 'long', distrib = rnd.exponential, d_options = {'scale': long_t2s[i]}, prevalence = long_A2s[i])
    col_dict.performElution()
    col_dict.gatherData()
    print(col_dict.sayCurrentRunDirectory())
    return 0

def dfm_runner(i, j, k, l, p, directory, mol_limit):

    # Need to create some hopping distribution arrays here:
    x = np.arange(20)
    hops = lambda p, x: np.exp(x*-p)
    hop_dist = hops(p, x)
    print(hop_dist)
    # Need to rewrite this line to correct the saving system.
    name = 'longSitedfm_'+str(i)+'x_len_'+str(j)+'adProb_'+str(k)+'prev_'+str(l)+'_bayesVal_'+str(p)
    print(name + ' ' + str(datetime.datetime.now()))
    stat_phase_dict = {'res': (int(j),5), 'contPDFs': True, 'hop_dist_array':hop_dist}
    col_dict = ChromaColumn(name = name, make_paths = False, stat_dict = stat_phase_dict, molecule_limit = mol_limit, cons_ads = k, directory = directory, competitive = True)
    col_dict.initConditions()
    col_dict.setUpMovie(framesPerFreeze = 250)
    col_dict.sayConditions(inDepth=True)
    col_dict.newSiteType(name = 'short', distrib = rnd.exponential, d_options = {'scale': 4.0}, prevalence = 1-l)
    col_dict.newSiteType(name = 'long', distrib = rnd.exponential, d_options = {'scale': 4.0*i}, prevalence = l)
    col_dict.performElution()
    col_dict.gatherData()
    #col_dict.dumpParameterstoJSON()
    print(col_dict.sayCurrentRunDirectory())
    return 0
    
def main(args):
    
    if args[1] == 'NicksData':
        i = int(args[2])
        mol_limit = int(args[3])
        directory = args[4]
        Nicks_c_runner(i, mol_limit, directory)

    elif args[1] == 'NicksData2':
        i = int(args[2])
        mol_limit = int(args[3])
        directory = args[4]
        Nicks_c_runner_2(i, mol_limit, directory)

    elif args[1] == 'NicksData3':
        i = int(args[2])
        j = int(args[3])
        k = float(args[4])
        l = float(args[5])
        directory = args[6]
        mol_limit = int(args[7])
        Nicks_c_runner_3(i, j, k, l, mol_limit, directory)
    
    elif args[1] == 'NicksData4':
        i = int(args[2])
        j = int(args[3])
        k = float(args[4])
        l = float(args[5])
        directory = args[6]
        mol_limit = int(args[7])
        Nicks_c_runner_4(i, j, k, l, mol_limit, directory)

    
    elif args[1] == 'DFM': # Numbering is off here, need to fix this.
        i = int(args[2])
        j = int(args[3])
        k = float(args[4])
        l = float(args[5])
        bayes_val = float(args[6])
        directory = args[7]
        mol_limit = int(args[8])


        dfm_runner(i, j, k, l, bayes_val, directory, mol_limit)
    
    elif len(args) == 8:
        i = int(args[1])
        j = int(args[2])
        k = float(args[3])
        l = float(args[4])
        directory = args[5]
        mol_limit = int(args[6])
        con_ads_len = float(args[7])
        c_runner(i, j, k, l, directory, mol_limit, con_ads_len)
if __name__ == "__main__":
    main(sys.argv)

"""
bash command to run the above

# Directory to save everything
directory="./saved_outputs/nicks_ads_chromatograms/cdf_only/"
con_ads_len=4
mol_limit=300000
for i in 3;
do
    for j in 500 1000 
    do
        for k in 0.2;
        do
            for l in 0.01 0.001; 
            do
            python column_runner.py NicksData4 $i $j $k $l $directory $mol_limit 
            done
        done
    done
done

directory="./saved_outputs/nicks_ads_chromatograms/cdf_only"
con_ads_len=4
mol_limit=300000
for i in 4 5 6;
do
    for j in 500 1000 
    do
        for k in 0.1 0.2 0.3 0.4 0.5;
        do
            for l in 0.01; 
            do
            python column_runner.py NicksData4 $i $j $k $l $directory $mol_limit 
            done
        done
    done
done

Code run on 06/05/2019.
directory="./saved_outputs/dfm_codes"
mol_limit=300000
for i in 5 25 50 125 250;
do
    for j in 1000 
    do
        for k in 0.5;
        do
            for l in 0.01; 
            do
                for p in 0.125 0.25 0.5 1 2 4;
                do
                python column_runner.py DFM $i $j $k $l $p $directory $mol_limit
                done
            done
        done
    done
done

directory="./saved_outputs/dfm_codes/100000/"
mol_limit=100000
for i in 5 25;
do
    for j in 250 500 750; 
    do
        for k in 0.25 0.5;
        do
            for l in 0.01; 
            do
                for p in 0.125 0.5 1 2 4;
                do
                python column_runner.py DFM $i $j $k $l $p $directory $mol_limit
                done
            done
        done
    done
done

# Adjusted code 10/07/2019
directory="./saved_outputs/dfm_codes_2/200000/"
mol_limit=200000
for i in 5 10 15 20;
do
    for j in 250 500 1000; 
    do
        for k in 0.25 0.5;
        do
            for l in 0.01 0.001; 
            do
                for p in 0.125 0.5 1 2 4;
                do
                python column_runner.py DFM $i $j $k $l $p $directory $mol_limit
                done
            done
        done
    done
done

directory="./saved_outputs/dfm_codes_2/200000/"
mol_limit=200000
for i in 45 50 55 60;
do
    for j in 250 500 1000; 
    do
        for k in 0.25 0.5;
        do
            for l in 0.01 0.001; 
            do
                for p in 0.125 0.5 1 2 4;
                do
                python column_runner.py DFM $i $j $k $l $p $directory $mol_limit
                done
            done
        done
    done
done

directory="./saved_outputs/dfm_codes_2/200000/"
mol_limit=200000
for i in 75 80 85 90
do
    for j in 250 500 1000; 
    do
        for k in 0.25 0.5;
        do
            for l in 0.01 0.001; 
            do
                for p in 0.125 0.5 1 2 4;
                do
                python column_runner.py DFM $i $j $k $l $p $directory $mol_limit
                done
            done
        done
    done
done

directory="./saved_outputs/dfm_codes_2/200000/"
mol_limit=200000
for i in 95 100 105 110
do
    for j in 250 500 1000; 
    do
        for k in 0.25 0.5;
        do
            for l in 0.01 0.001; 
            do
                for p in 0.125 0.5 1 2 4;
                do
                python column_runner.py DFM $i $j $k $l $p $directory $mol_limit
                done
            done
        done
    done
done
"""