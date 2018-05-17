# HTML (Hyper Text Markup Language)

```html
<!--HTML Notes-->

<!DOCTYPE html> <!--Describes the type of HTML, usually html, ...
                    which riggers Standards mode with all updated features-->
<html>
    <head>
        <!--meta information goes here-->
        <!--Describes meta information about the site ...
            and provides links to scripts and stylesheets the site needs ...
            to render and behave correctly.-->
        
        <!--the charset being used (the text's encoding): Unicode character-->
        <meta charset="utf-8"> 
        
        <!--the document's title (the text that shows up in browser tabs)-->
        <title>HTML Structure</title> 
        
        <!--associated CSS files (for style)-->
        <link rel="stylesheet" type="text/css" href="style.css"> 
        
        <!--associated JavaScript files (multipurpose scripts to change rendering and behavior)-->
        <script src="animations.js"></script> 
        
        <!--keywords, authors, and descriptions (often useful for SEO, search engine optimization)-->
        <meta name="description" content="This is what my website is all about!">
    </head>

    <body>
        <!--content goes here-->
        <!--Describes the actual content of the site that users will see.-->

        <!--heading-->
        <h1>This is a heading.</h1>
        <p><strong><em>author's name</em></strong></p>

        <!--div is used for grouping elements-->
        <div> 
            <p>This is a paragraph.</p>
            <span>This is a span.</span>
        </div>

        <!--figure-->
        <figure>
            <img src="image.jpg" alt="a picture" />
            <!--The alt attribute stands for "alternative description," which is important for people who use screen readers to browse the web. This is text that will show up in lieu of the actual image.-->
            <img src="http://somewebsite.com/image.jpg" alt="short description">
            <figcaption>This is a figure.</figcaption>
        </figure>
        
        <!--unordered list-->
        <ul> 
            <li>HTML</li>
            <li>CSS</li>
            <li>JavaScript</li>
        </ul>

        <!--table-->
        <table>
        <!-- An HTML table is defined as a series of rows (<tr>) -->
        <!-- The individual cell (<td>) contents are nested inside rows -->
        <!-- The <tr> tag is optional and is the parent of column headers (<th>) -->
            <tr>
                <th>First Header</th>
                <th>Second Header</th>
            </tr>
            <tr>
                <td>Row 2, Col 1</td>
                <td>Row 2, Col 2</td>
            </tr>
            <tr>
                <td>Row 3, Col 1</td>
                <td>Row 3, Col 2</td>
            </tr>
        </table>
        
        <!--hyperlink is created with anchor element. href is an attribute; content is what users see-->
        <a href="https://www.google.com">Google</a>

    </body>
</html>
```
<br>

### **Element, tag, and content**
each line in body is an element <br>
each <...> is a tag <br>
so each element usually contains opening tag, content, and closing tag <br>
image elements are called "void elements" <br>
<br>

### **HTML validator**
https://validator.w3.org/ <br>
analyze your website and verify that you're writing valid HTML

