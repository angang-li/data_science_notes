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


'''''''''''''''''''''''''''''
' Formatting with VBA
' color
    ' operates on cells(...) or range(...)
    .Font.ColorIndex = 1     ' font color black
    .Interior.ColorIndex = 6 ' cell color yellow
    .Interior.olor = vbGreen ' cell color green

    ' color index:
    ' https://msdn.microsoft.com/en-us/vba/excel-vba/articles/colorindex-property
    ' http://dmcritchie.mvps.org/excel/colors.htm

' number
    ' operates on cells(...) or range(...)
    .NumberFormat = "$#,##0.00" ' currency
    .Style = "Currency"         ' currency
    .NumberFormat = "0.00%"     ' percentage


'''''''''''''''''''''''''''''
' Common usages
' Loop through all worksheets in a workbook
    
    ' https://support.microsoft.com/en-us/help/142126/macro-to-loop-through-all-worksheets-in-a-workbook
    Dim currentSheet As Worksheet ' declare currentSheet as a worksheet object variable
    For Each currentSheet In Worksheets
	    ...
    Next

    Dim sheet1 As Worksheet
	Set sheet1 = Worksheets("Sheet1")
	sheet1.Cells(1,1).Value = "..."

' Concatenate ranges
    Range("A1:A" & 32).Value

' Find the last row, column, or cell in Excel
    ' https://www.excelcampus.com/vba/find-last-row-column-cell/
    last_row = Cells(Rows.Count, 1).End(xlUp).Row 'Find the last non-blank cell in column A(1)
    last_column = Cells(1, Columns.Count).End(xlToLeft).Column 'Find the last non-blank cell in row 1

    ' https://stackoverflow.com/questions/21268383/put-entire-column-each-value-in-column-in-an-array
    total_rows = currentSheet.Rows(currentSheet.Rows.Count).End(xlUp).Row

' Worksheet functions
	' Find the Row number of column I with 415 gunpowder
    match_gun = WorksheetFunction.Match(415, sheet1.Range("I2:I11"), 0)