import sys
import pandas as pd

def read_file(filename):

    speed_file = open(filename,'r')

    lines = speed_file.readlines()

    speed_file.close()

    return lines
def create_table(lines):
    
    
    table = {'time':[],'iter':[],'ls iter':[],'speedup':[]}

    index = []
    
    first_line = lines[0].split()
    #print first_line
    index.append(1)
    seq_time =float(first_line[1]) 
    table['time'].append(float(first_line[1]))
    table['iter'].append(int(first_line[2]))
    table['ls iter'].append(int(first_line[3]))
    table['speedup'].append(1)
    
    for line in lines[1:]:
        line_list = line.split()
        #print line_list
        index.append(line_list[1])
        par_time = float(line_list[2])
        table['time'].append(par_time)
        table['iter'].append(int(line_list[3]))
        table['ls iter'].append(int(line_list[4]))
        table['speedup'].append(seq_time/par_time)
    data = pd.DataFrame(table,index=index)
    
    return data
    
def main():
    try: 
        names = sys.argv[1:]
    except:
        print 'Give file'
        return
    for name in names:
        lines = read_file(name)
        data = create_table(lines)
        print data
        #data.to_latex('latexTables/'+name[:-3]+'tex')
if __name__ == '__main__':
    main()
