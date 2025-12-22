"""
PDF Report Generator for Dental AI Analysis
Creates professional clinical reports from AI analysis results
"""
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PIL import Image
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def generate_pdf_report(
    original_image: Image.Image,
    annotated_image: Optional[Image.Image],
    detections: List[Dict],
    ai_analysis: Dict[str, str],
    patient_info: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a professional PDF report from dental X-ray analysis
    
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
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
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
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=8,
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
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#e74c3c'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    # ============ HEADER ============
    story.append(Paragraph("Dental X-Ray Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Information Section
    if patient_info:
        patient_data = [
            ["Patient Name:", patient_info.get("name", "N/A")],
            ["Patient ID:", patient_info.get("id", "N/A")],
            ["Date of Birth:", patient_info.get("dob", "N/A")],
            ["Date of Examination:", patient_info.get("exam_date", datetime.now().strftime("%Y-%m-%d"))]
        ]
    else:
        patient_data = [
            ["Patient Name:", "Placeholder - Not for Clinical Use"],
            ["Patient ID:", "N/A"],
            ["Date of Birth:", "N/A"],
            ["Date of Examination:", datetime.now().strftime("%Y-%m-%d")]
        ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7'))
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ============ IMAGES SECTION ============
    story.append(Paragraph("X-Ray Images", heading_style))
    
    # Save images to temporary files for PDF
    temp_images = []
    
    # Original Image
    temp_original = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    original_image.save(temp_original.name, 'PNG')
    temp_images.append(temp_original.name)
    
    original_img = RLImage(temp_original.name, width=5*inch, height=4*inch)
    story.append(Paragraph("Original X-Ray", subheading_style))
    story.append(original_img)
    story.append(Spacer(1, 0.2*inch))
    
    # Annotated Image (if available)
    if annotated_image:
        temp_annotated = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        annotated_image.save(temp_annotated.name, 'PNG')
        temp_images.append(temp_annotated.name)
        
        annotated_img = RLImage(temp_annotated.name, width=5*inch, height=4*inch)
        story.append(Paragraph("Annotated X-Ray with Detections", subheading_style))
        story.append(annotated_img)
        story.append(Spacer(1, 0.2*inch))
    else:
        # Use original if no annotated image
        story.append(Paragraph("Note: No annotations available", subheading_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(PageBreak())
    
    # ============ FINDINGS TABLE ============
    story.append(Paragraph("Detection Findings", heading_style))
    
    if detections and len(detections) > 0:
        # Create findings table
        findings_data = [["#", "Class", "Location", "Confidence"]]
        
        for idx, detection in enumerate(detections, 1):
            class_name = detection.get("class_name", detection.get("description", "Unknown"))
            position = detection.get("position", "N/A")
            confidence = detection.get("confidence", 0.0)
            confidence_str = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else str(confidence)
            
            findings_data.append([
                str(idx),
                class_name,
                position.replace("-", " ").title(),
                confidence_str
            ])
        
        findings_table = Table(findings_data, colWidths=[0.5*inch, 2*inch, 2*inch, 1.5*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        story.append(findings_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Summary statistics
        total_detections = len(detections)
        class_counts = {}
        for det in detections:
            class_name = det.get("class_name", "Unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary_text = f"<b>Summary:</b> {total_detections} detection(s) found. "
        summary_text += ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
        story.append(Paragraph(summary_text, body_style))
    else:
        story.append(Paragraph("No dental features detected in this X-ray.", body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # ============ AI ANALYSIS SECTION ============
    story.append(Paragraph("AI Analysis Summary", heading_style))
    
    # GPT-4o-mini Analysis
    if ai_analysis.get("gpt4"):
        story.append(Paragraph("GPT-4o-mini Analysis", subheading_style))
        gpt_text = ai_analysis["gpt4"].replace("\n", "<br/>")
        story.append(Paragraph(gpt_text, body_style))
        story.append(Spacer(1, 0.15*inch))
    
    # Llama 3.3 Analysis
    if ai_analysis.get("groq"):
        story.append(Paragraph("Llama 3.3 70B Analysis", subheading_style))
        groq_text = ai_analysis["groq"].replace("\n", "<br/>")
        story.append(Paragraph(groq_text, body_style))
        story.append(Spacer(1, 0.15*inch))
    
    # Qwen 3 32B Analysis (formerly Mixtral)
    if ai_analysis.get("mixtral"):
        story.append(Paragraph("Qwen 3 32B Analysis", subheading_style))
        mixtral_text = ai_analysis["mixtral"].replace("\n", "<br/>")
        story.append(Paragraph(mixtral_text, body_style))
        story.append(Spacer(1, 0.15*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # ============ TREATMENT RECOMMENDATIONS ============
    story.append(Paragraph("Treatment Recommendations", heading_style))
    
    # Generate recommendations based on detections
    recommendations = []
    
    if detections:
        for det in detections:
            class_name = det.get("class_name", "").lower()
            if "impacted" in class_name:
                recommendations.append(
                    "• Impacted wisdom tooth detected. Consider consultation with oral surgeon for extraction evaluation."
                )
            elif "cavity" in class_name or "caries" in class_name:
                recommendations.append(
                    "• Dental caries detected. Recommend dental restoration treatment and follow-up care."
                )
            elif "implant" in class_name:
                recommendations.append(
                    "• Dental implant present. Monitor for stability and proper osseointegration."
                )
    
    if not recommendations:
        recommendations.append(
            "• No immediate treatment recommendations. Continue regular dental check-ups."
        )
    
    for rec in recommendations:
        story.append(Paragraph(rec, body_style))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # ============ FOOTER ============
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("─" * 80, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Report Generated: {timestamp}", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    disclaimer_text = (
        "<b>IMPORTANT DISCLAIMER:</b><br/>"
        "This report is generated by an AI system for educational and research purposes only. "
        "It is NOT intended for clinical diagnosis or medical decision-making. "
        "All findings must be verified by licensed dental professionals. "
        "This system does not replace professional dental consultation or clinical judgment. "
        "No medical decisions should be made based solely on this report."
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
            model_responses = entry.get("model_responses", {})
            # Check if this was a vision analysis
            if any(key in model_responses for key in ["gpt4", "groq", "mixtral"]):
                # Try to find detections in the response
                # This would need to be stored during processing
                pass
        
        if entry.get("role") == "user" and entry.get("image"):
            original_image = entry["image"]
            break
    
    return detections, original_image, annotated_image

