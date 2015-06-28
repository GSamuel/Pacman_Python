import os, re, glob, shutil, subprocess, sys, getopt, pickle
from keyboard import keyboard
import pacman
import imp

solutionModule='competitionAgents'
solutionFile  =solutionModule + '.py'
execList=['-p', 'MyPacmanAgent', '-q', '-t', '--frameTime', '0', '-f', '-n', '50']
studentRegexp=r'_([^_]*)_.*'
logfileName = 'output.txt'
htmlstyle = 'leaderboard.css'
htmloutputfile = 'leaderboard.html'

def saveobject(filename,obj):
  #print "Saving ",obj
  #print "to %s\n" % filename
  output = open(filename, 'wb')
  pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def loadobject(filename):
  input = open(filename,'rb');
  obj=pickle.load(input)
  return obj

def getGroupName(file):
   m = re.search(studentRegexp, file)
   if m==None:
      m = re.search(studentRegexp[1:], file[file.find('/')+1:])
   return m.group(1) if not m==None else None

def gamesStats(games):
   scores = [game.state.getScore() for game in games]
   timeouts = [game.state.getTimeout() for game in games]
   moves  = [max(1,game.numMoves) for game in games]
   times  = [games[i].totalAgentTimes[0]/float(moves[i]) for i in range(len(games))]
   wins = [game.state.isWin() for game in games]
   winRate = wins.count(True)/ max(1,float(len(wins)))
   avescore = sum(scores) / max(1,float(len(scores)))
   return (scores,timeouts,moves,times,wins,winRate,avescore)


def generateHtml(games, name=None):
   if type(games[0]) is list :
      scores = []; timeouts=[]; moves=[]; times=[]; wins=[]; winRate=[]; avescore=[];
      for laygames in games: # average the per-layout averages
         (lscores,ltimeouts,lmoves,ltimes,lwins,lwinRate,lavescore)=gamesStats(laygames)
         scores.append(   sum(lscores)  /float(max(1,len(lscores)))   )
         timeouts.append( sum(ltimeouts)/float(max(1,len(ltimeouts))) )
         moves.append(    sum(lmoves)   /float(max(1,len(lmoves)))    )
         times.append(    sum(ltimes)   /float(max(1,len(ltimes)))    )
         wins.append(     sum(lwins)    /float(max(1,len(lwins)))     )
         avescore.append(lavescore)
      avescore = sum(avescore) / float(max(1,len(avescore)))
   else:
      (scores,timeouts,moves,times,wins,winRate,avescore)=gamesStats(games)

   if name == None: name='Unnamed'
   htmlout = ''
   htmlout += ''.join('<tr class="alt"><th rowspan="4">%s</th>' % name)
   htmlout += '<td rowspan="4" class="score">%d</td>' % avescore
   htmlout += '<th class="alt">Score</th>'
   htmlout += ' '.join(["<td>%d</td>" % score for score in scores])
   htmlout += '</tr>\n'
   htmlout += '<tr><th class="alt">Win/Timeout</th>'
   htmlout += ' '.join(["<td>%.1f/%.1f</td>" % (wins[i],timeouts[i]) for i in range(len(wins))])
   htmlout += '<td></td></tr>\n'
   htmlout += '<tr><th class="alt">Moves</th>'
   htmlout += ' '.join(["<td>%d</td>" % move for move in moves])
   htmlout += '</tr>\n'
   htmlout += '<tr><th class="alt">Move Time</th>'
   htmlout += ' '.join(["<td>%.3f</td>" % time for time in times])
   htmlout += '</tr>\n'
   return (avescore,htmlout)

def runFile(file,replay=False,args=[]):
  failed = True
  score  = 0  
  htmlsummary= ''
  fileoutput = ''
  games=None
  # Fetch student number
  print(file)
  studentNumber = getGroupName(file)
  if studentNumber is None: studentNumber=''
  print('Trying ' + studentNumber + '\nfrom file: ' + file) 
  fileoutput += studentNumber + ':' + '\n'
  #keyboard()

  # Move their competitionAgents.py to the environment
  if not file == solutionFile :
    try:
      shutil.copyfile(file,solutionFile)
    except shutil.SameFileError as err: # samefile isn't a bug
      pass
    except IOError as err:
      print('Error couldnt copy over the solution file: ' + file)
      return (score,htmlsummary)

  # Run the environment
  output = ''
  failed = False

  # force reload of student code -- needed even for load to get the class definitions correctly
  try :
    print(sys.modules[solutionModule])
    imp.reload(sys.modules[solutionModule]) 
  except:
    failed=True
    output = 'Error could not load solution module';
    print(output)
    
  if not failed :
      if os.path.isfile(studentNumber + '_games.pk') and not replay:
        games = loadobject(studentNumber + '_games.pk')
      else:
        try:
          nargs = execList;
          print('Running with arguments :' + '  '.join(execList + args) )
          games = pacman.cmdlineRunGames(execList + args)
        except subprocess.CalledProcessError as e:          
          failed = True
          output = e.output
      if not failed:
        try:
          saveobject(studentNumber + '_games.pk',games) # save so don't have to re-run later
        except:
          print('Couldnt pickel the saved games...')
        score,htmlsummary = generateHtml(games,studentNumber)    

  # Print result
  if failed:
    fileoutput += 'Failed :\n' + output + '\n\n'
  else:
    fileoutput += 'Output :\n' + output + '\n\n'
  print(fileoutput)
  return (score,htmlsummary)

def help():
   return """Usage: ./{filename} [-h | --help --dir --num] [directory] [studentNumber]

This script copies the search.py and searchAgents.py pairwise from the given directory and executes pacman.py with all 12 configurations to test the search algorithms. 
The output is dumped into output.txt.
If this script runs with the studentNumber argument, it will only run search.py and searchAgents.py of that one student.
If no studentNumber argument is given, it will run all pairs of search.py and searchAgents.py.

-h|--help\t\tDisplays this help text.
--dir=[directory]\tSets the default directory with student solutions to the entered one. By default it is `{defaultDir}`.
--num=[studentNumber]\tScript will only execute search.py and searchAgents.py from this student group.

Examples:
./{filename}\t\t\tRuns all file pairs in {defaultDir}/
./{filename} --dir=solutions\tRuns all file pairs in solutions/
./{filename} --num=s4050614\t\tRuns the file pair in {defaultDir}/ of the student 4050614
""".format(filename=__file__, defaultDir='studentsolutions')

def main(argv):
  defaultDirectory = 'studentsolutions'
  studentNumber = None
  replay=False
  args=None

  # Remove the old fileoutput file
  if os.path.isfile(logfileName):
     shutil.copy(logfileName,logfileName + '.bak')
     os.remove(logfileName)
  logfile = open(logfileName,'w');

  # Parse the arguments
  opts=[]
  try:
     opts, args = getopt.getopt(argv, 'h', ['help', 'dir=', 'num=', 'replay'])
  except getopt.GetoptError:
    _, e, _ = sys.exc_info()
    if e.msg.find('not recognized')<0: # pass through unrecog options
       print((help()))
       sys.exit(2)

  for opt, arg in opts:
    if opt in ('-h', '--help'):
      print((help()))
      sys.exit()
    elif opt == '--dir':
       defaultDirectory = arg
    elif opt == '--num':
       studentNumber = arg
    elif opt == '--replay':
       replay = True       

  print(args)
  fileoutput = ''

  if not os.path.isdir(defaultDirectory):
    print(("Error : could not find directory :", defaultDirectory))
    exit(2)
  
  info=list()
  if studentNumber is None:
    # If you want to run them all at once
    solnfiles = sorted(glob.glob(defaultDirectory + '/*' + solutionFile))
    # extract student number and remove duplicates
    oldgroup=None
    files=[]
    #print('\n'.join(solnfiles))
    if len(solnfiles)==1:
      files=solnfiles
    else:
      for file in solnfiles:
        groupName = getGroupName(file)
        if not groupName == None :
          if groupName == oldgroup:
            files[-1]=file
          else:
            oldgroup = groupName
            files.append(file)
    print('\n'.join(files))
    for file in files:
      score,output = runFile(file,replay,args)
      print(output)
      logfile.write(output)
      info.append((score,output))
  else:
    # If you want to run one at the time using the student number
    for file in glob.glob(defaultDirectory + '/*' + solutionFile):
      groupName = getGroupName(file)
      if 'Groep ' + studentNumber == groupName or ( groupName is None and studentNumber=='' ):
        score,output = runFile(file,replay,args)
        print(output)
        logfile.write(output)
        info.append((score,output))

  logfile.close()
  # sort the results based on score
  info = sorted(info,reverse=True)
  shutil.copy(htmlstyle,htmloutputfile) # copy in style info
  hfile = open(htmloutputfile,'a')
  # write the header
  hfile.write('<table border="1">')
  hfile.write("\n<tr><th>Group</th><th>Ave Score</th><th>Info/Level</th>")
  hfile.write(''.join(["<th>%d</th>" % i for i in range(12)]))
  hfile.write("\n")
  for score,output in info:
     hfile.write(output)
  # write the footer
  hfile.write("</table>");
  hfile.close()
  return info

if __name__ == "__main__":
  # make back up of current solution file
  if os.path.isfile(solutionFile):
    shutil.copy(solutionFile,solutionFile+ ".bak");
  if not os.path.isfile(solutionFile+".orig"):
    shutil.copy(solutionFile,solutionFile + ".orig")  
  try:
    __import__(solutionModule)
  except :
    print("Couldnt import the file " + solutionFile)
    print("copy back from orginal and re-try:" + solutionFile + ".orig")
    sys.exit(-1)
      
  # ask for arguments if none given
  if len(sys.argv) == 1:
    if sys.version_info < (3,0):
      print("Python version <3.0 detected, YMMV")
      sys.argv.extend([x for x in re.split(r' *',raw_input("Enter any command line arguments?")) if x!=''])
    else:
      sys.argv.extend([x for x in re.split(r' *',input("Enter any command line arguments?")) if x!=''])
  main(sys.argv[1:])

  # copy the orginal file back
  #if os.path.isfile(solutionFile + '.orig'):
  #  shutil.copy(solutionFile + ".orig",solutionFile) # restore backup
