from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 8)
pdf.write(txt="Text ", h=5)
# задаем цвет текста
pdf.set_text_color(255, 0, 0)
pdf.write(txt="red color", h=5)
# новая строка
pdf.ln(15)
# отключаем стиль текста
pdf.set_font(family="Arial", style="B", size=18)
pdf.write(txt="Globally", h=5)
# Для отключения красного цвета
# нужно задать черный цвет
pdf.set_text_color(0)
# новая строка
pdf.ln(25)
pdf.set_font_size(23)
pdf.write(txt="Now", h=5)
pdf.ln(15)
# цвет заливки - желтый
pdf.set_fill_color(255, 255, 0)
pdf.set_font(family="Arial", style="I", size=18)
pdf.write(txt="Good ", h=5)
# текст для выделения выведем в ячейке
pdf.cell(txt="Morning", w=5, fill=True)
pdf.write(txt="in Python", h=5)
pdf.output("style.pdf")
