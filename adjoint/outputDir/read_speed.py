import sys
import pandas as pd

def read_file(filename):

    speed_file = open(filename,'r')

    lines = speed_file.readlines()

    speed_file.close()

    return lines
def create_table(lines,with_iter=True,with_iter2=False,with_norm=False,ideal_S=False):
    
    if with_iter:
        table = {'time':[],'iter':[],'ls iter':[],'speedup':[]}
    elif with_iter2:
        table = {'time':[],'iter':[],'ls iter':[],'speedup':[],'gr':[],'fu':[]}
    elif with_norm:
        table = {'time':[],'iter':[],'ls iter':[],'speedup':[],'gr':[],'fu':[],'norm':[]}
        with_iter2 =True
    elif ideal_S:
        table = {'time':[],'L':[],'speedup':[],'S2':[],'norm':[],'f':[]}
    else:
        table = {'time':[],'speedup':[]}
    index = []
    
    first_line = lines[0].split()
    #print first_line
    index.append(1)
    seq_time =float(first_line[1])
    seq_fugr =1
    table['time'].append(float(first_line[1]))
    if with_iter:
        table['iter'].append(int(first_line[2]))
        table['ls iter'].append(int(first_line[3]))
    elif with_iter2:
        table['iter'].append(int(first_line[4]))
        table['ls iter'].append(int(first_line[5]))
        table['gr'].append(int(first_line[3]))
        table['fu'].append(int(first_line[2]))
        if with_norm:
            table['norm'].append('--')
    elif ideal_S:
        table['L'].append(int(first_line[3])+int(first_line[2]))
        table['S2'].append(1)
        table['norm'].append('--')
        table['f'].append('--')
        seq_fugr = float(int(first_line[3])+int(first_line[2]))
    table['speedup'].append(1)
    
    for line in lines[1:]:
        line_list = line.split()
        #print line_list 
        index.append(line_list[1])
        par_time = float(line_list[2])
        table['time'].append(par_time)
        if with_iter:
            table['iter'].append(int(line_list[3]))
            table['ls iter'].append(int(line_list[4]))
        elif with_iter2:
            table['iter'].append(int(line_list[5]))
            table['ls iter'].append(int(line_list[6]))
            table['gr'].append(int(line_list[4]))
            table['fu'].append(int(line_list[3]))
            if with_norm:
                table['norm'].append(float(line_list[7]))
        elif ideal_S:
            table['L'].append(int(first_line[4])+int(first_line[3]))
            table['S2'].append(int(line_list[1][:-1])*seq_fugr/(int(first_line[4])+int(first_line[3])))
            table['norm'].append(float(line_list[7]))
            table['f'].append(float(line_list[9]))
        table['speedup'].append(seq_time/par_time)
    #print table
    data = pd.DataFrame(table,index=index)
    return data
    
def main():
    try:
        if sys.argv[1] == '0':
            names = sys.argv[2:]
            with_iter = False
            with_iter2 = False
            with_norm =False
            ideal_S=False
        elif sys.argv[1] == '1':
            names = sys.argv[2:]
            with_iter = False
            with_iter2 = True
            with_norm = False
            ideal_S=False
        elif sys.argv[1] == '2':
            names = sys.argv[2:]
            with_iter = False
            with_iter2 = False
            with_norm =True
            ideal_S=False
        elif sys.argv[1] == '3':
            names = sys.argv[2:]
            with_iter = False
            with_iter2 = False
            with_norm =False
            ideal_S=True
        else:
            names = sys.argv[1:]
            with_iter = True
            with_iter2 = False
            with_norm =False
            ideal_S=False
    except:
        print 'Give file'
        return
    for name in names:
        print name
        lines = read_file(name)
        data = create_table(lines,with_iter=with_iter,with_iter2=with_iter2,
                            with_norm=with_norm,ideal_S=ideal_S)
        print data
        #data.to_latex('latexTables/'+name[:-3]+'tex')
if __name__ == '__main__':
    main()
