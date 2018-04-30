# **Git and Github**
Notes taken from [Udacity](https://www.udacity.com/course/how-to-use-git-and-github--ud775)'s online course

<!-- TOC -->

- [**Git and Github**](#git-and-github)
- [Navigating a commit history](#navigating-a-commit-history)
    - [Find the difference using Mac](#find-the-difference-using-mac)
    - [View history using Git](#view-history-using-git)
    - [Find difference using Git](#find-difference-using-git)
    - [Statistics of which files have changed in each commit](#statistics-of-which-files-have-changed-in-each-commit)
    - [Clone Git repository](#clone-git-repository)
    - [Getting Colored Output](#getting-colored-output)
    - [Checkout an old version](#checkout-an-old-version)
- [Creating and modifying a repository](#creating-and-modifying-a-repository)
    - [Show hidden files](#show-hidden-files)
    - [Initialize a new git repository, there is no commit yet](#initialize-a-new-git-repository--there-is-no-commit-yet)
    - [Show which file has been changed since last commit](#show-which-file-has-been-changed-since-last-commit)
    - [Add files to the staging area, aka., changes to be committed](#add-files-to-the-staging-area--aka--changes-to-be-committed)
    - [Remove a file from staging area](#remove-a-file-from-staging-area)
    - [Commit changes and write commit message](#commit-changes-and-write-commit-message)
    - [Show any changes in the working directory that have not been added to the staging area yet](#show-any-changes-in-the-working-directory-that-have-not-been-added-to-the-staging-area-yet)
    - [Show changes in the staging area that have not been committed yet](#show-changes-in-the-staging-area-that-have-not-been-committed-yet)
    - [Discard any change from both the working directory and the staging area](#discard-any-change-from-both-the-working-directory-and-the-staging-area)
    - [Branches are helpful](#branches-are-helpful)
    - [Show the current branch](#show-the-current-branch)
    - [Create a new branch](#create-a-new-branch)
    - [Checkout the new branch](#checkout-the-new-branch)
    - [Or create and checkout in one command](#or-create-and-checkout-in-one-command)
    - [See the visual representation of the commit history](#see-the-visual-representation-of-the-commit-history)
    - [Delete previous branch](#delete-previous-branch)
    - [Merge branch2 and branch3 into master branch](#merge-branch2-and-branch3-into-master-branch)
    - [Show the difference added by 1 particular commit without knowing its parent](#show-the-difference-added-by-1-particular-commit-without-knowing-its-parent)

<!-- /TOC -->

<br>

# Navigating a commit history
## Find the difference using Mac
```
diff -u old_game.js new_game.js
```
`-u`: unified diff format, making the output easier to read

## View history using Git
```
git log
```
Shows id number, author, date, and commit message

## Find difference using Git
```
git diff <id_number_1> <id_number_2>
```

## Statistics of which files have changed in each commit
```
git log --stat
```
Shows ++++-- across multiple files <br>
Press `q` to exit git log

## Clone Git repository
```
git clone <url>
```

## Getting Colored Output
To get colored `diff` output, run <br>
```
git config --global color.ui auto
```

## Checkout an old version
```
git checkout <id_number>
```

<br>

# Creating and modifying a repository
## Show hidden files
```
ls -a
```

## Initialize a new git repository, there is no commit yet
```
git init
```

## Show which file has been changed since last commit
```
git status
```

## Add files to the staging area, aka., changes to be committed
```
git add filename.txt
```

## Remove a file from staging area
```
git reset filename.txt
```

## Commit changes and write commit message
```
git commit
```
Or
```
git commit -m "Commit message"
```
Commit message style guide: http://udacity.github.io/git-styleguide/

## Show any changes in the working directory that have not been added to the staging area yet
```
git diff
```

## Show changes in the staging area that have not been committed yet
```
git diff --staged
```

## Discard any change from both the working directory and the staging area
```
git reset --hard
```

## Branches are helpful 
* e.g. production quality vs development
* e.g. unique feature
* e.g. experimental work

## Show the current branch
```
git branch
```

## Create a new branch
```
git branch new_branch_name
```

## Checkout the new branch
```
git checkout new_branch_name
```
## Or create and checkout in one command
```
git checkout -b new_branch_name
```

## See the visual representation of the commit history 
```
git log --graph --oneline branch1 branch2
```

## Delete previous branch
```
git branch -d branch_name
```

If a branch is deleted and leaves some commits unreachable from existing branches, 
those commits will continue to be accessible by commit id, until Git’s garbage collection runs. You can also run this process manually with `git gc`.

## Merge branch2 and branch3 into master branch
```
git checkout master
git merge branch2 branch3
```

## Show the difference added by 1 particular commit without knowing its parent
```
git show commit_id
```

