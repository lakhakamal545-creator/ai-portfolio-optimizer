import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(tickers, weights, sharpe, ret, vol, max_dd, filename="Portfolio_Report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30, textColor=colors.HexColor("#0A2342"))
    sub_style = ParagraphStyle('CustomSub', parent=styles['Heading2'], fontSize=14, spaceAfter=20, textColor=colors.HexColor("#C5A059"))
    
    elements = []
    elements.append(Paragraph("AI Portfolio Optimization Report", title_style))
    elements.append(Paragraph("Generated for: Kamal Lakha | MBA Financial Markets", styles['Italic']))
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Risk & Performance Metrics", sub_style))
    metrics_data = [["Metric", "Value"], ["Expected Annual Return", f"{ret:.2%}"], ["Annual Volatility (Risk)", f"{vol:.2%}"], ["Sharpe Ratio", f"{sharpe:.2f}"], ["Maximum Drawdown", f"{max_dd:.2%}"]]
    t = Table(metrics_data, colWidths=[200, 100])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (1,0), colors.HexColor("#0A2342")), ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8f9fa")), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    elements.append(t)
    elements.append(Spacer(1, 30))
    
    elements.append(Paragraph("Optimized Asset Allocation", sub_style))
    alloc_data = [["Ticker", "Target Weight"]]
    for tick, w in zip(tickers, weights):
        if w > 0.001: alloc_data.append([tick, f"{w:.2%}"])
            
    t_alloc = Table(alloc_data, colWidths=[150, 150])
    t_alloc.setStyle(TableStyle([('BACKGROUND', (0,0), (1,0), colors.HexColor("#C5A059")), ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    elements.append(t_alloc)
    
    doc.build(elements)
    return filename
