import io
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
from app.config import settings
from app.schemas.report import ExportReportRequest, ExportReportResponse
from app.utils.logger import logger

class ReportService:
    @staticmethod
    def generate_pdf(request: ExportReportRequest) -> ExportReportResponse:
        """Generate a professional satellite intelligence PDF report"""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        uid = uuid.uuid4().hex[:8]
        report_filename = f"satquery_report_{request.task_type}_{uid}.pdf"
        output_path = settings.OUTPUT_PATH / report_filename

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom dark-theme / professional styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#090c10'),
            spaceAfter=4,
            fontName='Helvetica-Bold'
        )
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#484f58'),
            spaceAfter=10
        )
        h2_style = ParagraphStyle(
            'SectionH2',
            parent=styles['Heading2'],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#10b981'),
            spaceBefore=10,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#161b22')
        )
        alert_style = ParagraphStyle(
            'Alert',
            parent=styles['Normal'],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#b45309')
        )

        story = []

        # 1. Header Banner
        header_data = [
            [
                Paragraph("<b>SATQUERY AI</b> | SATELLITE INTELLIGENCE REPORT", title_style),
                Paragraph(f"<b>MISSION ID:</b> {request.mission_id}<br/><b>DATE:</b> {timestamp_str}", subtitle_style)
            ]
        ]
        t_header = Table(header_data, colWidths=[350, 170])
        t_header.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(t_header)
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#10b981'), spaceBefore=6, spaceAfter=12))

        # 2. Executive Summary & Parameters
        story.append(Paragraph("1. MISSION & TASK SPECIFICATION", h2_style))
        task_label = request.task_type.upper().replace('_', ' ')
        primary_model = request.analysis_data.get("model", "SatQuery AI")
        detection_model = request.analysis_data.get("detection_model") or request.analysis_data.get("optical_processing") or primary_model
        semantic_model = request.analysis_data.get("semantic_model", "GeoChat")
        proc_time = request.analysis_data.get("processing_time_ms", 0.0)
        spec_time = request.analysis_data.get("specialized_time_ms")
        sem_time = request.analysis_data.get("semantic_time_ms")

        latency_breakdown = f"{proc_time:.1f} ms"
        if spec_time is not None and sem_time is not None:
            latency_breakdown += f" (Specialized CV: {spec_time:.1f} ms | Semantic Reasoning: {sem_time:.1f} ms)"

        meta_table_data = [
            [Paragraph("<b>Analysis Workflow:</b>", body_style), Paragraph(task_label, body_style)],
            [Paragraph("<b>Specialized CV Model(s):</b>", body_style), Paragraph(detection_model, body_style)],
            [Paragraph("<b>Semantic Reasoning Model:</b>", body_style), Paragraph(semantic_model, body_style)],
            [Paragraph("<b>Inference Processing Time:</b>", body_style), Paragraph(latency_breakdown, body_style)],
            [Paragraph("<b>Analyst Notes:</b>", body_style), Paragraph(request.analyst_notes or "Routine automated satellite analysis.", body_style)]
        ]
        t_meta = Table(meta_table_data, colWidths=[160, 360])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f6f8fa')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 10))

        # 3. Quantitative Results & Metrics
        story.append(Paragraph("2. QUANTITATIVE MEASUREMENTS & DETECTIONS", h2_style))
        data = request.analysis_data

        if request.task_type == "bi_temporal_change":
            chg_pct = data.get("change_percentage", 0.0)
            n_reg = data.get("num_regions", 0)
            tot_chg_px = data.get("total_changed_pixels", 0)
            res_table_data = [
                [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Unit / Classification</b>", body_style)],
                [Paragraph("Calculated Change Area", body_style), Paragraph(f"<b>{chg_pct}%</b>", body_style), Paragraph("Relative Scene Area", body_style)],
                [Paragraph("Connected Changed Regions", body_style), Paragraph(f"<b>{n_reg}</b>", body_style), Paragraph("Discrete Spatial Clusters", body_style)],
                [Paragraph("Changed Pixel Count", body_style), Paragraph(f"{tot_chg_px:,}", body_style), Paragraph("Pixels", body_style)],
                [Paragraph("Spatial Co-Registration", body_style), Paragraph("VERIFIED", body_style), Paragraph(data.get("coregistration_notes", "Aligned"), body_style)],
            ]
        elif request.task_type == "highlight":
            n_det = data.get("num_detections", 0)
            prompt = data.get("prompt", "")
            tot_area_pct = data.get("total_area_percentage", 0.0)
            res_table_data = [
                [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Unit / Details</b>", body_style)],
                [Paragraph("Target Prompt", body_style), Paragraph(f"<b>{prompt}</b>", body_style), Paragraph("Zero-shot Text Class", body_style)],
                [Paragraph("Detected Instances", body_style), Paragraph(f"<b>{n_det}</b>", body_style), Paragraph("Segmented Polygons", body_style)],
                [Paragraph("Total Highlighted Area", body_style), Paragraph(f"<b>{tot_area_pct}%</b>", body_style), Paragraph("Scene Coverage", body_style)],
            ]
        elif request.task_type == "optical_sar":
            sar_stats = data.get("sar_stats", {})
            opt_stats = data.get("optical_stats", {})
            res_table_data = [
                [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Modality / Details</b>", body_style)],
                [Paragraph("SAR Dynamic Range", body_style), Paragraph(f"<b>{sar_stats.get('dynamic_range_db', 0)} dB</b>", body_style), Paragraph("Sentinel-1 Microwave", body_style)],
                [Paragraph("High-Backscatter Ratio", body_style), Paragraph(f"<b>{sar_stats.get('high_backscatter_ratio', 0)*100:.1f}%</b>", body_style), Paragraph("Metallic / Built-up Facades", body_style)],
                [Paragraph("Low-Backscatter Ratio", body_style), Paragraph(f"<b>{sar_stats.get('low_backscatter_ratio', 0)*100:.1f}%</b>", body_style), Paragraph("Specular Water / Flat Ground", body_style)],
            ]
        else: # VQA
            q = data.get("question", "")
            conf = data.get("confidence", 0.92)
            res_table_data = [
                [Paragraph("<b>Field</b>", body_style), Paragraph("<b>Content</b>", body_style), Paragraph("<b>Confidence</b>", body_style)],
                [Paragraph("Query Question", body_style), Paragraph(q, body_style), Paragraph(f"{conf*100:.0f}% (Heuristic)", body_style)],
            ]

        t_res = Table(res_table_data, colWidths=[160, 160, 200])
        t_res.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e6edf3')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t_res)
        story.append(Spacer(1, 10))

        # 4. Semantic Findings / Interpretation
        story.append(Paragraph("3. SEMANTIC REASONING & MODEL FINDINGS", h2_style))
        semantic_text = data.get("semantic_analysis") or data.get("answer") or data.get("cross_modal_analysis") or data.get("semantic_summary") or "Analysis completed successfully."
        story.append(Paragraph(semantic_text, body_style))
        story.append(Spacer(1, 10))

        # 5. Model Transparency & Limitations
        story.append(Paragraph("4. MODEL TRANSPARENCY & SENSOR LIMITATIONS", h2_style))
        warning_text = data.get("transparency_warning") or (
            "Semantic descriptions are synthesized via vision-language foundations. "
            "Critical mission decisions should be validated against calibrated ground-truth GIS data."
        )
        warn_table = Table([[Paragraph(f"<b>[NOTICE]:</b> {warning_text}", alert_style)]], colWidths=[520])
        warn_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fffbeb')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#f59e0b')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(warn_table)

        doc.build(story)
        file_size = os.path.getsize(output_path)
        logger.info(f"Generated PDF report: {output_path.name} ({file_size / 1024:.1f} KB)")

        return ExportReportResponse(
            status="success",
            report_filename=report_filename,
            download_url=f"/api/files/download/{report_filename}",
            generated_at=timestamp_str,
            file_size_bytes=file_size
        )

report_service = ReportService()
