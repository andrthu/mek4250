import sys

val = sys.argv[1]
name = sys.argv[2]

string = '''
print %s

name = '%s'
print name
''' %(val,name)

new_script = open(name,'w')

print string
new_script.write(string)
new_script.close()
