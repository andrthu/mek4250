import sys

val = sys.argv[1]
name = sys.argv[2]

string = '''
print %s

name = '%s'
print name
''' %(val,name)

string2 = """
while [ $((10-$COUNTER2)) -gt 0 ]; do
	mpiexec python func_grad_speedup_test.py %s 1 1 $val;
	let COUNTER2=COUNTER2+1;
done
mpiexec python func_grad_speedup_test.py %s 1 0 $val;
mpiexec python func_grad_speedup_test.py %s 1 2 $val;

rm temp_time.txt;
""" %(val,val,val)
print string2
new_script = open(name,'w')


new_script.write(string)
new_script.close()
