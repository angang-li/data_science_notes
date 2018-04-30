###########################
Navigating a commit history
###########################

Find difference using Mac
diff -u old_game.js new_game.js
-u: Unified diff format, making the output easier to read

View history using Git
git log
Shows serial number, author, date, and commit message

Find difference using Git
git diff serial_number_1 serial_number_2

Statistics of which files have changed in each commit
git log --stat
Shows ++++-- across multiple files
Press q to exit git log

Clone Git repository
git clone <url>

Getting Colored Output
To get colored diff output, run git config --global color.ui auto

Checkout an old version
git checkout serial_number 

###################################
Creating and modifying a repository
###################################

Show hidden files
ls -a

Initialize a new git repository, there is no commit yet
git init

Show which file has been changed since last commit
git status

Add files to the staging area, aka., changes to be committed
git add filename.txt

Remove a file from staging area
git reset filename.txt

Commit changes and write commit message
git commit                           or
git commit -m "Commit message"
commit message style guide: http://udacity.github.io/git-styleguide/

Show any changes in the working directory that have not been added to the staging area yet
git diff

Show changes in the staging area that have not been committed yet
git diff --staged

Discard any change from both the working directory and the staging area
git reset --hard

Branches are helpful 
e.g. production quality vs development
e.g. unique feature
e.g. experimental

Show the current branch
git branch

Create a new branch
git branch new_branch_name
Checkout the new branch
git checkout new_branch_name
Or do the two thing in one command
git checkout -b new_branch_name

See the visual representation of the commit history 
git log --graph --oneline branch1 branch2

If a branch is deleted and leaves some commits unreachable from existing branches, 
those commits will continue to be accessible by commit id, until Git’s garbage collection runs. 
This will happen automatically from time to time, unless you actively turn it off. 
You can also run this process manually with git gc.

Merge branch 2 and 3 into master branch
git checkout master
git merge branch2 branch3

Show the difference added by 1 particular commit without knowing its parent
git show commit_id

Delete previous branch
git branch -d branch_name