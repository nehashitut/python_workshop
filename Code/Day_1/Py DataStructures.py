# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:18:00 2016

@author: Gunnvant
"""

## Python Data Structures
#- Ints and floats
#- Str
#- Lists
#- Tuples
#- Dictionaries
#- Files

### Ints and Floats
#- Depending upon the version of python used the behaviour of ints and floats can differ, we will be using python 2.7

a = 4
print a
b= 4.0
print b


# We can know the type of the object by using the type() method
print type(a)
print type(b)

a1=3
a2=6
print a1/a2
print type(a1/a2)

a1=3
a2=6
print a1/float(a2)
print type(a1/float(a2))

#We can know about the methods available in the respective python classes by using the dir() method
print dir(a)
print dir(b)

### Strings
#- Strings as iterables
#- Slicing and indexing
#- String methods

st='Jigsaw'
print st[0]
print st[1]
print st[0:4]
print st[-1]

print dir(st)

print len(st)
print st.count('i')
print st.find('i')
print st.find('l')

w="Two Shōkaku-class aircraft carriers, Shōkaku and Zuikaku, were commissioned by the Imperial Japanese Navy during World War II. They participated in the attack on Pearl Harbor, the Indian Ocean Raid, and the battles of the Coral Sea, the Eastern Solomons, and the Santa Cruz Islands. Their air groups sank two of the four fleet carriers lost by the United States Navy during the war in addition to one elderly British light carrier. Returning to Japan after the Battle of the Coral Sea to repair damage and replace lost aircraft, they missed the Battle of Midway in June 1942. After the catastrophic loss of four carriers during that battle, they formed the bulk of Japan's carrier force for the rest of the war. Shōkaku was sunk by an American submarine during the Battle of the Philippine Sea in June 1944 as the Americans invaded the Mariana Islands, and Zuikaku was sacrificed as part of a decoy force four months later in the Battle of Leyte Gulf, both with heavy loss of life. Historian Mark Peattie called them 'arguably the best aircraft carriers' of the early 1940s."

print w

### Looping and conditionals interlude
#- Using for loops and if-else statements

# All iterables can be looped through
for c in w[0:20]:
      print c
     
# We can control the flow of execution of code by using conditionals such as if-else
if 2>3: #Notice that a statement is followed by a :
    print 'Oh boy!!!'
else:
    print 'Yay!!'
    
    
### Class Assignment
# Use the assign_txt
# Count number of capital letters
# How many times has the word RSS occured?
# How many times has the word Modi occured?
    
assign_txt="Narendra Damodardas Modi, born 17 September 1950) is the 15th and current Prime Minister of India, in office since 26 May 2014.odi, a leader of the Bharatiya Janata Party was the Chief Minister of Gujarat from 2001 to 2014 and is the Member of Parliament from Varanasi. He led the BJP in the 2014 general election, which gave the party a majority in the Lok Sabha, the first for any political party in India since 1984.As the Chief Minister of Gujarat, Modi's economic policies were praised, while his administration was also criticised for failing to significantly improve the human development in the state, and for failing to prevent the 2002 Gujarat riots. A Hindu nationalist and member of the Rashtriya Swayamsevak Sangh, Modi,remains a controversial figure domestically and internationally. Modi was born on 17 September 1950, to a family of grocers in Vadnagar, Mehsana district, Bombay State (present-day Gujarat).Modi's family belonged to the Modh-Ghanchi-Teli (oil-presser) community,hich is categorised as an Other Backward Class by the Indian government.Modi was the third of six children born to Damodardas Mulchand (1915–1989) and Heeraben Modi (b. c. 1920).As a child, Modi helped his father sell tea at the Vadnagar railway station, and later ran a tea stall with his brother near a bus terminus.Modi completed his higher secondary education in Vadnagar in 1967, where a teacher described him as an average student and a keen debater, with an interest in theatre.Modi had an early gift for rhetoric in debates, and this was noted by his teachers and students.Modi preferred playing larger-than-life characters in theatrical productions, which has influenced his political image.At age eight, Modi discovered the Rashtriya Swayamsevak Sangh (RSS), and began attending its local shakhas (training sessions). There, Modi met Lakshmanrao Inamdar, popularly known as Vakil Saheb, who inducted him as an RSS balswayamsevak (junior cadet) and became his political mentor.While Modi was training with the RSS, he also met Vasant Gajendragadkar and Nathalal Jaghda, Bharatiya Jana Sangh leaders who were founding members of the BJP's Gujarat unit in 1980.Engaged while still a child to a local girl, Jashodaben Narendrabhai Modi, Modi rejected the arranged marriage at the same time he graduated from high school.The resulting familial tensions contributed to his decision to leave home in 1967.Modi spent the ensuing two years travelling across Northern and North-eastern India, though few details of where he went have emerged.In interviews, Modi has described visiting Hindu ashrams founded by Swami Vivekananda: the Belur Math near Kolkata, followed by the Advaita Ashrama in Almora and the Ramakrishna mission in Rajkot. Modi remained only a short time at each, since he lacked the required college education."


###  Lists
# Lists as buckets of data
# List slicing
# List methods
# List comprehension
# Use of map, lambdas, filter and reduce

#Lists are the most commonly used data structure
l1=[1,2,3,4,5,6]
print l1
print l1[0]
print l1[0:3]

print dir(l1)

l1.append(7)
print l1

### Class assignment
# Use the assign_txt
# Find out the total number of lines in assign_txt (Use the split() method to create a list, assume "." to be EOL)
# Extract the years like 1950 from the text and count their occurence

#### List comprehension
# Its a concise way of creating lists
# Usefull convention to know about

a=[1,2,3,4]
b=['a','b','c','d']

# Suppose we want to create a list from a and b such that it looks like the following:
# [[1,'a'],[2,'b'],[3,'c'],[4,'d']]

[[x,y] for x in a for y in b if a.index(x)==b.index(y)]

c=range(11)
print c

#Suppose we want to create a list containing the squares in the list c
#Method 1
s=[]
for num in c:
    s.append(num**2)
print s

#Method 2
[x**2 for x in c ] #Very concise way

### Lambdas, map, filter and reduce
# We can use lambdas, filter, map and reduce to replicate the functionality of list comprehension

mass=[45,55,65,76]
ht=[1.65,1.70,1.55,1.80]
bmi=map(lambda x,y:x/y**2,mass,ht)
bmi_f=filter(lambda x:x<20,bmi)
print bmi
print bmi_f
su=reduce(lambda x,y:x+y,bmi)
print su

#Using list comprehension to create equivalent results
#bmi
BMI=[x/y**2 for x in mass for y in ht if mass.index(x)==ht.index(y)]
print BMI
#filter
[x for x in BMI if x<20]


### Tuples
# Tuples as immutable iterables
# Tuple methods

t=(1,2,3,5)
print t

print t[0]
print t[1]

li6=[1,2,3,4]
li6[0]=0
print li6

t[0]=0

print dir(t)

# zip
a=[1,2,3,4]
b=['a','b','c','d']
zip(a,b)

[list(x) for x in zip(a,b) ]

### Dictionaries
# Dictionaries as hashbles
# Extracting keys, values
# Sorting  a dictionary
# Looping through a dictionary

di={'Abi_wt':100, 'Abi_sleeps': 16}
print di
print dir(di)

print di.values()
print di.keys()

print di.items()

sorted(di.iteritems())

di['Abi_sleeps']=20
print di

'Abi_wt' in di
20 in di#We can check if particular keys exist using simple 'in' statement

for x in di:
    print x
    
for x in di:
    print x, di[x]
    
### Use the assign_txt
# Find out the occurence of each word in text string (Create a list of each word in the string, loop over this list to create a dictionary of words and their occurences
    
### Files
# Reading files
# Manipulating data in file objects
# Writing out files
    
# modules interlude
import os
os.chdir('F:\\Work\\Python')

f=open('tweets_assignment.txt','r')
fo=f.read()
print type(f)
print type(fo)
f.close()

f=open('write.txt','w')
for dt in fo:
    f.write(dt+'\n')
f.close()

fo[0:100]

### Assignment
# Use the file tweets_assignment.txt
# Count the total number of tweets in the file
# Extract the time stamp from the file and store it in a list

