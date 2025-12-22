"""
Professional PDF Report Generator for Dental AI Analysis
Creates clinical-grade reports from AI analysis results
"""
import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.lib.utils import ImageReader


# ============ HELPER FUNCTIONS ============

def get_tooth_number_fdi(position: str) -> str:
    """Convert position to FDI tooth notation"""
    position_lower = position.lower()
    if "upper-left" in position_lower or "upper right" in position_lower:
        quadrant = "1" if "left" in position_lower else "2"
    elif "lower-left" in position_lower or "lower right" in position_lower:
        quadrant = "3" if "left" in position_lower else "4"
    else:
        return "N/A"
    
    # For wisdom teeth, typically 18, 28, 38, 48
    # This is simplified - real FDI would need more precise positioning
    if "wisdom" in position_lower or "third molar" in position_lower:
        return f"{quadrant}8"
    return f"{quadrant}X"  # Generic if not wisdom tooth


def calculate_severity(class_name: str, confidence: float) -> Tuple[str, str]:
    """Calculate severity and priority based on detection"""
    class_lower = class_name.lower()
    
    if "impacted" in class_lower:
        if confidence > 0.8:
            severity = "Severe"
            priority = "Immediate"
        elif confidence > 0.6:
            severity = "Moderate"
            priority = "Soon"
        else:
            severity = "Mild"
            priority = "Routine"
    elif "caries" in class_lower or "deep caries" in class_lower:
        if "deep" in class_lower:
            severity = "Severe"
            priority = "Immediate"
        else:
            severity = "Moderate"
            priority = "Soon"
    elif "lesion" in class_lower:
        severity = "Moderate"
        priority = "Soon"
    else:
        severity = "Mild"
        priority = "Routine"
    
    return severity, priority


def calculate_risk_score(detections: List[Dict]) -> Tuple[str, str, float]:
    """Calculate overall risk assessment score"""
    if not detections:
        return "Low", "green", 0.2
    
    high_risk_count = 0
    moderate_risk_count = 0
    
    for det in detections:
        class_name = det.get("class_name", "").lower()
        confidence = det.get("confidence", 0.5)
        
        if "impacted" in class_name and confidence > 0.7:
            high_risk_count += 1
        elif "deep caries" in class_name:
            high_risk_count += 1
        elif "caries" in class_name or "lesion" in class_name:
            moderate_risk_count += 1
    
    total_risk = high_risk_count * 2 + moderate_risk_count
    
    if total_risk >= 3:
        return "High", "red", 0.9
    elif total_risk >= 1:
        return "Medium", "orange", 0.6
    else:
        return "Low", "green", 0.3


def get_recommended_action(class_name: str, severity: str) -> str:
    """Get specific recommended action (shortened for table display)"""
    class_lower = class_name.lower()
    
    if "impacted" in class_lower:
        if severity == "Severe":
            return "Extraction consult (2 weeks)"
        elif severity == "Moderate":
            return "Extraction consult (1 month)"
        else:
            return "Monitor, routine follow-up"
    elif "caries" in class_lower:
        if "deep" in class_lower:
            return "Immediate restoration required"
        else:
            return "Restoration (1-2 months)"
    elif "lesion" in class_lower:
        return "Further diagnostic imaging"
    else:
        return "Routine monitoring"


# ============ CUSTOM PAGE TEMPLATE ============

def draw_header_footer(canvas, doc):
    """Draw professional header and footer on each page"""
    canvas.saveState()
    
    # Get report ID from doc
    report_id = getattr(doc, 'report_id', 'N/A')
    
    # Header background (blue)
    canvas.setFillColor(colors.HexColor('#2c3e50'))
    canvas.rect(0, letter[1] - 0.75*inch, letter[0], 0.75*inch, fill=1, stroke=0)
    
    # Clinic name
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(0.75*inch, letter[1] - 0.4*inch, "DENTAL AI CLINIC")
    
    # Logo placeholder
    canvas.setFillColor(colors.HexColor('#95a5a6'))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.75*inch, letter[1] - 0.6*inch, "[LOGO PLACEHOLDER]")
    
    # Report ID
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(letter[0] - 0.75*inch, letter[1] - 0.4*inch, f"Report ID: {report_id}")
    
    # Footer background
    canvas.setFillColor(colors.HexColor('#ecf0f1'))
    canvas.rect(0, 0, letter[0], 0.5*inch, fill=1, stroke=0)
    
    # Page number
    canvas.setFillColor(colors.HexColor('#2c3e50'))
    canvas.setFont("Helvetica", 9)
    page_num = canvas.getPageNumber()
    canvas.drawCentredString(letter[0]/2, 0.25*inch, f"Page {page_num}")
    
    # Disclaimer
    canvas.setFillColor(colors.HexColor('#7f8c8d'))
    canvas.setFont("Helvetica-Oblique", 7)
    canvas.drawCentredString(letter[0]/2, 0.1*inch, "AI-Generated Report - For Educational Purposes Only")
    
    canvas.restoreState()


# ============ MAIN REPORT GENERATOR ============

def generate_pdf_report(
    original_image: Image.Image,
    annotated_image: Optional[Image.Image],
    detections: List[Dict],
    ai_analysis: Dict[str, str],
    patient_info: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a professional clinical PDF report from dental X-ray analysis
    
    Args:
        original_image: Original X-ray image
        annotated_image: Annotated image with bounding boxes (optional)
        detections: List of detection dictionaries with class, position, confidence
        ai_analysis: Dictionary with model responses (gpt4, groq, mixtral)
        patient_info: Optional patient information dict
        output_path: Optional output file path (if None, creates temp file)
    
    Returns:
        Path to generated PDF file
    """
    # Create output file if not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(temp_dir, f"dental_report_{timestamp}.pdf")
    
    # Generate report ID
    report_id = str(uuid.uuid4())[:8].upper()
    
    # Create PDF document with custom template
    class CustomDocTemplate(BaseDocTemplate):
        def __init__(self, filename, report_id, **kwargs):
            BaseDocTemplate.__init__(self, filename, **kwargs)
            self.report_id = report_id
            
            # Create page template with header/footer
            frame = Frame(
                self.leftMargin,
                self.bottomMargin,
                self.width,
                self.height,
                id='normal'
            )
            template = PageTemplate(id='main', frames=frame, onPage=draw_header_footer)
            self.addPageTemplates([template])
    
    doc = CustomDocTemplate(
        output_path,
        report_id=report_id,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1.25*inch,  # Extra space for header
        bottomMargin=0.75*inch
    )
    
    # Container for PDF elements
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold',
        backColor=colors.HexColor('#3498db'),
        borderPadding=8,
        borderWidth=1,
        borderColor=colors.HexColor('#2980b9')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    highlight_style = ParagraphStyle(
        'Highlight',
        parent=body_style,
        backColor=colors.HexColor('#fff3cd'),
        borderPadding=6,
        borderWidth=1,
        borderColor=colors.HexColor('#ffc107')
    )
    
    # ============ TITLE PAGE ============
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("DENTAL X-RAY ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information Section with color coding
    if patient_info:
        patient_data = [
            ["Patient Name:", patient_info.get("name", "N/A")],
            ["Patient ID:", patient_info.get("id", "N/A")],
            ["Date of Birth:", patient_info.get("dob", "N/A")],
            ["Date of Examination:", patient_info.get("exam_date", datetime.now().strftime("%Y-%m-%d"))],
            ["Report ID:", report_id],
            ["Report Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
    else:
        patient_data = [
            ["Patient Name:", "Placeholder - Not for Clinical Use"],
            ["Patient ID:", "N/A"],
            ["Date of Birth:", "N/A"],
            ["Date of Examination:", datetime.now().strftime("%Y-%m-%d")],
            ["Report ID:", report_id],
            ["Report Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
    
    patient_table = Table(patient_data, colWidths=[2.5*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2980b9')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 0.4*inch))
    
    # ============ RISK ASSESSMENT ============
    risk_level, risk_color, risk_value = calculate_risk_score(detections)
    risk_colors_map = {
        "red": colors.HexColor('#e74c3c'),
        "orange": colors.HexColor('#f39c12'),
        "green": colors.HexColor('#27ae60')
    }
    
    risk_data = [
        ["RISK ASSESSMENT", risk_level.upper()],
    ]
    
    risk_table = Table(risk_data, colWidths=[4*inch, 2.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#2c3e50')),
        ('BACKGROUND', (1, 0), (1, 0), risk_colors_map.get(risk_color, colors.HexColor('#95a5a6'))),
        ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
    ]))
    
    story.append(risk_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ============ DENTAL HEALTH SUMMARY ============
    story.append(Paragraph("Dental Health Summary", heading_style))
    
    summary_points = []
    if detections:
        class_counts = {}
        for det in detections:
            class_name = det.get("class_name", "Unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            severity, _ = calculate_severity(class_name, 0.7)
            if severity == "Severe":
                summary_points.append(f"<b>⚠️ {count} {class_name}(s) detected</b> - Requires immediate attention")
            elif severity == "Moderate":
                summary_points.append(f"<b>• {count} {class_name}(s) detected</b> - Schedule follow-up")
            else:
                summary_points.append(f"• {count} {class_name}(s) detected - Routine monitoring")
    else:
        summary_points.append("• No significant dental issues detected in this X-ray")
        summary_points.append("• Continue regular dental check-ups")
    
    for point in summary_points:
        story.append(Paragraph(point, body_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # ============ IMAGES SECTION (SIDE-BY-SIDE) ============
    story.append(Paragraph("X-Ray Images", heading_style))
    
    # Save images to temporary files
    temp_images = []
    temp_original = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    original_image.save(temp_original.name, 'PNG')
    temp_images.append(temp_original.name)
    
    original_img = RLImage(temp_original.name, width=3.2*inch, height=2.5*inch)
    
    if annotated_image:
        temp_annotated = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        annotated_image.save(temp_annotated.name, 'PNG')
        temp_images.append(temp_annotated.name)
        annotated_img = RLImage(temp_annotated.name, width=3.2*inch, height=2.5*inch)
        
        # Side-by-side images
        image_data = [
            ["Original X-Ray", "Annotated X-Ray with Detections"],
            [original_img, annotated_img]
        ]
    else:
        image_data = [
            ["Original X-Ray", ""],
            [original_img, ""]
        ]
    
    image_table = Table(image_data, colWidths=[3.5*inch, 3.5*inch])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 11),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor('#2c3e50')),
        ('BOTTOMPADDING', (0, 0), (1, 0), 8),
        ('TOPPADDING', (0, 0), (1, 0), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
    ]))
    
    story.append(image_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ============ DETAILED FINDINGS TABLE ============
    story.append(Paragraph("Detailed Findings", heading_style))
    
    if detections and len(detections) > 0:
        findings_data = [["#", "Tooth #", "Condition", "Severity", "Confidence", "Action", "Priority"]]
        
        # Calculate column widths based on proportions (total ~6.5 inches available)
        # # : 5%, Tooth # : 10%, Condition : 15%, Severity : 10%, Confidence : 12%, Action : 35%, Priority : 13%
        total_width = 6.5 * inch
        col_widths = [
            0.05 * total_width,  # # (5%)
            0.10 * total_width,  # Tooth # (10%)
            0.15 * total_width,  # Condition (15%)
            0.10 * total_width,  # Severity (10%)
            0.12 * total_width,  # Confidence (12%)
            0.35 * total_width,  # Action (35%)
            0.13 * total_width,  # Priority (13%)
        ]
        
        # Store colors for each row
        row_colors = []
        
        for idx, detection in enumerate(detections, 1):
            class_name = detection.get("class_name", detection.get("description", "Unknown"))
            position = detection.get("position", "N/A")
            confidence = detection.get("confidence", 0.0)
            tooth_num = get_tooth_number_fdi(position)
            severity, priority = calculate_severity(class_name, confidence)
            action = get_recommended_action(class_name, severity)
            
            # Color code severity
            severity_color = colors.HexColor('#27ae60')  # Green
            if severity == "Severe":
                severity_color = colors.HexColor('#e74c3c')  # Red
            elif severity == "Moderate":
                severity_color = colors.HexColor('#f39c12')  # Orange
            
            # Color code priority
            priority_color = colors.HexColor('#95a5a6')  # Gray
            if priority == "Immediate":
                priority_color = colors.HexColor('#e74c3c')  # Red
            elif priority == "Soon":
                priority_color = colors.HexColor('#f39c12')  # Orange
            
            confidence_str = f"{confidence:.0%}"
            
            # Use Paragraph for Action column to enable text wrapping
            action_para = Paragraph(action, ParagraphStyle(
                'ActionStyle',
                parent=body_style,
                fontSize=8,
                alignment=TA_LEFT,
                leading=10
            ))
            
            findings_data.append([
                str(idx),
                tooth_num,
                class_name,
                severity,
                confidence_str,  # Just percentage, no progress bar
                action_para,  # Paragraph for text wrapping
                priority
            ])
            
            # Store colors for this row
            row_colors.append((severity_color, priority_color))
        
        findings_table = Table(findings_data, colWidths=col_widths, repeatRows=1)
        
        # Build table style with all color coding
        table_style_commands = [
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (4, 0), 'CENTER'),  # Center align first 5 columns
            ('ALIGN', (5, 0), (5, 0), 'LEFT'),   # Left align Action header
            ('ALIGN', (6, 0), (6, 0), 'CENTER'), # Center align Priority header
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),  # Add left padding
            ('RIGHTPADDING', (0, 0), (-1, -1), 8), # Add right padding
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (0, 1), (4, -1), 'CENTER'),  # Center align first 5 columns
            ('ALIGN', (5, 1), (5, -1), 'LEFT'),    # Left align Action column for better readability
            ('ALIGN', (6, 1), (6, -1), 'CENTER'),  # Center align Priority
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),  # Add bottom padding to data rows
            ('TOPPADDING', (0, 1), (-1, -1), 8),     # Add top padding to data rows
        ]
        
        # Add color coding for severity and priority columns for each row
        for row_idx, (severity_color, priority_color) in enumerate(row_colors, start=1):
            table_style_commands.append(('TEXTCOLOR', (3, row_idx), (3, row_idx), severity_color))  # Severity column
            table_style_commands.append(('TEXTCOLOR', (6, row_idx), (6, row_idx), priority_color))  # Priority column
        
        findings_table.setStyle(TableStyle(table_style_commands))
        
        story.append(findings_table)
        story.append(Spacer(1, 0.2*inch))
    else:
        story.append(Paragraph("No dental features detected in this X-ray.", body_style))
    
    story.append(PageBreak())
    
    # ============ AI ANALYSIS SECTION ============
    story.append(Paragraph("AI Analysis Summary", heading_style))
    
    # Combine all AI analyses into one comprehensive summary
    all_analyses = []
    if ai_analysis.get("gpt4"):
        all_analyses.append(("GPT-4o-mini", ai_analysis["gpt4"]))
    if ai_analysis.get("groq"):
        all_analyses.append(("Llama 3.3 70B", ai_analysis["groq"]))
    if ai_analysis.get("mixtral"):
        all_analyses.append(("Qwen 3 32B", ai_analysis["mixtral"]))
    
    if all_analyses:
        for model_name, analysis_text in all_analyses:
            story.append(Paragraph(f"{model_name} Analysis", subheading_style))
            analysis_clean = analysis_text.replace("\n", "<br/>")
            story.append(Paragraph(analysis_clean, body_style))
            story.append(Spacer(1, 0.15*inch))
    else:
        story.append(Paragraph("No AI analysis available.", body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # ============ PATIENT-FRIENDLY EXPLANATION ============
    story.append(Paragraph("Patient-Friendly Explanation", heading_style))
    
    explanation_text = """
    <b>What This Report Means:</b><br/><br/>
    This report analyzes your dental X-ray using advanced AI technology. The system has identified specific areas 
    in your X-ray that may require attention. Here's what you need to know:<br/><br/>
    
    <b>• Detections Found:</b> The AI system has highlighted areas in your X-ray where dental features were identified. 
    Each detection includes a confidence level showing how certain the system is about the finding.<br/><br/>
    
    <b>• Risk Level:</b> Your overall risk assessment is based on the types and number of findings. 
    This helps prioritize which issues need immediate attention versus routine monitoring.<br/><br/>
    
    <b>• Next Steps:</b> Based on the findings, specific recommendations are provided. These should be discussed 
    with your dentist, who can provide personalized treatment plans based on your complete dental history.<br/><br/>
    
    <b>Important:</b> This report is a tool to help your dentist, but it does not replace professional clinical judgment. 
    Always consult with your dentist for diagnosis and treatment decisions.
    """
    
    story.append(Paragraph(explanation_text, highlight_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ============ NEXT STEPS CHECKLIST ============
    story.append(Paragraph("Next Steps Checklist", heading_style))
    
    checklist_items = []
    if detections:
        high_priority = [d for d in detections if calculate_severity(d.get("class_name", ""), d.get("confidence", 0))[1] == "Immediate"]
        if high_priority:
            checklist_items.append("☐ Schedule urgent dental consultation within 2 weeks")
        
        moderate_priority = [d for d in detections if calculate_severity(d.get("class_name", ""), d.get("confidence", 0))[1] == "Soon"]
        if moderate_priority:
            checklist_items.append("☐ Schedule dental appointment within 1 month")
        
        checklist_items.append("☐ Review this report with your dentist")
        checklist_items.append("☐ Discuss treatment options and timeline")
        checklist_items.append("☐ Follow recommended preventive care")
    else:
        checklist_items.append("☐ Continue regular dental check-ups")
        checklist_items.append("☐ Maintain good oral hygiene")
        checklist_items.append("☐ Schedule routine 6-month examination")
    
    for item in checklist_items:
        story.append(Paragraph(item, body_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # ============ TREATMENT RECOMMENDATIONS (PRIORITIZED) ============
    story.append(Paragraph("Prioritized Treatment Recommendations", heading_style))
    
    recommendations = []
    if detections:
        # Group by priority
        immediate = []
        soon = []
        routine = []
        
        for det in detections:
            class_name = det.get("class_name", "")
            severity, priority = calculate_severity(class_name, det.get("confidence", 0.5))
            action = get_recommended_action(class_name, severity)
            
            rec_text = f"<b>{class_name}</b> - {action}"
            
            if priority == "Immediate":
                immediate.append(rec_text)
            elif priority == "Soon":
                soon.append(rec_text)
            else:
                routine.append(rec_text)
        
        if immediate:
            story.append(Paragraph("<b>🔴 IMMEDIATE ACTION REQUIRED:</b>", subheading_style))
            for rec in immediate:
                story.append(Paragraph(f"• {rec}", body_style))
                story.append(Spacer(1, 0.1*inch))
        
        if soon:
            story.append(Paragraph("<b>🟠 SCHEDULE SOON:</b>", subheading_style))
            for rec in soon:
                story.append(Paragraph(f"• {rec}", body_style))
                story.append(Spacer(1, 0.1*inch))
        
        if routine:
            story.append(Paragraph("<b>🟢 ROUTINE MONITORING:</b>", subheading_style))
            for rec in routine:
                story.append(Paragraph(f"• {rec}", body_style))
                story.append(Spacer(1, 0.1*inch))
    else:
        story.append(Paragraph("• No immediate treatment recommendations. Continue regular dental check-ups.", body_style))
    
    story.append(PageBreak())
    
    # ============ PROFESSIONAL FOOTER SECTION ============
    story.append(Spacer(1, 0.5*inch))
    
    # QR Code placeholder
    qr_placeholder = Table([["QR Code Placeholder", "Scan for digital verification"]], colWidths=[2*inch, 4*inch])
    qr_placeholder.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#7f8c8d')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
    ]))
    story.append(qr_placeholder)
    story.append(Spacer(1, 0.3*inch))
    
    # Signature line
    signature_data = [
        ["Reviewed by:", "_________________________", "Date:", "_________________________"],
        ["", "Dentist Name", "", datetime.now().strftime("%Y-%m-%d")]
    ]
    signature_table = Table(signature_data, colWidths=[1.5*inch, 2.5*inch, 1*inch, 2*inch])
    signature_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Oblique'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(signature_table)
    story.append(Spacer(1, 0.3*inch))
    
    # System information
    system_info = [
        ["AI System Version:", "Dental AI Platform v2.3"],
        ["Report Generation Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Report ID:", report_id],
    ]
    system_table = Table(system_info, colWidths=[2*inch, 4.5*inch])
    system_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#7f8c8d')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ecf0f1')),
    ]))
    story.append(system_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Final disclaimer
    disclaimer_text = (
        "<b>IMPORTANT DISCLAIMER:</b><br/><br/>"
        "This report is generated by an AI system for educational and research purposes only. "
        "It is NOT intended for clinical diagnosis or medical decision-making. "
        "All findings must be verified by licensed dental professionals. "
        "This system does not replace professional dental consultation or clinical judgment. "
        "No medical decisions should be made based solely on this report. "
        "For clinical use, this report must be reviewed and signed by a licensed dentist."
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=body_style,
        fontSize=8,
        textColor=colors.HexColor('#e74c3c'),
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique',
        backColor=colors.HexColor('#ffe6e6'),
        borderPadding=10
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build PDF
    doc.build(story)
    
    # Clean up temporary image files
    for temp_img in temp_images:
        try:
            os.unlink(temp_img)
        except:
            pass
    
    return output_path


def extract_detections_from_state(conversation_state: List) -> Tuple[List[Dict], Optional[Image.Image], Optional[Image.Image]]:
    """
    Extract detection data from conversation state
    
    Returns:
        (detections, original_image, annotated_image)
    """
    detections = []
    original_image = None
    annotated_image = None
    
    # Find most recent vision analysis
    for entry in reversed(conversation_state):
        if entry.get("role") == "assistant":
            if entry.get("yolo_detections"):
                detections = entry.get("yolo_detections", [])
            if entry.get("original_image"):
                original_image = entry.get("original_image")
            if entry.get("annotated_image"):
                annotated_image = entry.get("annotated_image")
            if detections or original_image:
                break
        
        if entry.get("role") == "user" and entry.get("image") and not original_image:
            original_image = entry["image"]
    
    return detections, original_image, annotated_image
