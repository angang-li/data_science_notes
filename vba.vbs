' VBA
' visual basic applications

'''''''''''''''''''''''''''''
' Programming building blocks
' variables / arrays
    dim var1 as string  ' variable declaration
    var1 = "peanut"     ' variable assignment

    dim arr1(2) as string
    arr1(0) = ""
    arr1(1) = ""
    arr1(2) = ""

    dim arr2() as string ' a dynamic way to define array with unknown length
    arr2 = split("lap top", " ")

    ' variable types: string, integer, double, long, boolean
    ' casting: convert variable types, e.g., str(age)

' conditionals
    if ... then
        ...
    elseif ... then
        ...
    else
        ...
    end if

    ' operators: and, or, <>

' iterations
    ' nested for loop
    dim i as integer
    dim j as integer
    for i = 1 to 3
        for j = 1 to 5
            ...
        next j
    next i

' functions
    sub function_name():
        msgbox("Hello!") ' message box
        Cells(2,1).Value = "string in the A2 cell" ' cell
		Range("A1").Value = "string in the A1 cell" ' range
		Range("A3", "C3").Value = "something else" ' whole thing extends from A3 to B3
		Range("A3, C3").Value = "something else" ' whole thing only in A3 and C3
		Range("A4:H4").Value = "something else"
    end sub



