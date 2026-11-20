import html
import math
import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from analysis.ideal_ranges import IDEAL_RANGES


def is_finite_number(value) -> bool:
    try:
        if isinstance(value, bool) or value is None:
            return False
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def get_effective_joint_range(joint, batting_stance):
    """Return the same reference range used by calculate_score()."""
    reference_joint = joint
    stance = str(batting_stance or "").strip().lower()

    if stance in ("left-handed", "left handed"):
        if joint.startswith("left_"):
            reference_joint = "right_" + joint[5:]
        elif joint.startswith("right_"):
            reference_joint = "left_" + joint[6:]

    return IDEAL_RANGES.get(reference_joint)


def get_correction_advice(mistake):
    """Return general practice guidance for recognized technique flaws."""
    text = str(mistake or "").strip().lower()

    if "front elbow" in text and "low" in text:
        return (
            "Practise controlled shadow batting while keeping the "
            "front elbow comfortably raised and controlled."
        )

    if "front knee" in text and "straight" in text:
        return (
            "Practise a balanced batting stance with a slight, "
            "comfortable front-knee bend. Avoid locking the knee."
        )

    if "front shoulder" in text and "dropping" in text:
        return (
            "Practise slow shadow swings while maintaining a stable "
            "head position and controlled front shoulder."
        )

    return (
        "Review this movement across repeated video frames and, "
        "where possible, compare it with guidance from a cricket coach."
    )


def clean_pdf_text(text) -> str:
    """Strip LaTeX math syntax, normalize symbols, and escape XML characters for ReportLab."""
    if text is None:
        return ""
    cleaned = (
        str(text)
        .replace("$", "")
        .replace("\\times", "×")
        .replace("^{\\circ}", "°")
        .replace("^{o}", "°")
        .replace("&deg;", "°")
    )
    return html.escape(cleaned, quote=False)


def generate_pdf_report(data: dict, output_pdf_path: str = None) -> str:
    """Generate a PDF performance report directly from an analysis summary dictionary."""
    if output_pdf_path is None:
        video_name = data.get("video_name", "analysis")
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        output_pdf_path = f"{base_name}_performance_report.pdf"

    # Setup Document Page Geometry
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    story = []

    # Color Palette Definitions
    PRIMARY = colors.HexColor("#1E3A8A")
    TEXT_COLOR = colors.HexColor("#1F2937")
    BG_LIGHT = colors.HexColor("#F8FAFC")
    BORDER_COLOR = colors.HexColor("#CBD5E1")

    # Styles Setup
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
    )
    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=TEXT_COLOR,
    )
    bold_body_style = ParagraphStyle(
        "BoldBodyText",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    footwork_feedback_style = ParagraphStyle(
        "FootworkFeedback",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-10,
        leading=14,
        spaceAfter=5,
    )

    # Document Header
    video_name = clean_pdf_text(data.get("video_name", "Unknown Video"))
    date_str = datetime.now().strftime("%B %d, %Y - %H:%M")

    story.append(Paragraph("🏏 AI Cricket Performance Report", title_style))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"<b>Source Video:</b> {video_name} &nbsp;|&nbsp; <b>Generated:</b> {date_str}",
            body_style,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=PRIMARY,
            spaceBefore=0,
            spaceAfter=12,
        )
    )

    # ============================================================
    # SECTION 1: EXECUTIVE SUMMARY
    # ============================================================
    story.append(Paragraph("1. Executive Summary", h2_style))

    source_info = data.get("source_video", {}) or {}
    final_report = data.get("final_report", {}) or {}
    impact_events = data.get("impact_events", []) or []
    has_impact = len(impact_events) > 0

    declared_valid = bool(data.get("analysis_valid", has_impact))
    analysis_valid = declared_valid and has_impact
    validity_mismatch = declared_valid != has_impact

    final_score = data.get("final_score")
    technique_score = final_report.get("technique_score")
    shot_status = str(data.get("shot_status", "") or "").strip()
    shot_status_lower = shot_status.lower()

    # Shot Type: use reviewed reference label when present, otherwise fall back to final_shot
    human_annotation = data.get("human_reviewed_annotation") or {}
    reviewed_shot_type = None
    if (
        isinstance(human_annotation, dict)
        and human_annotation.get("status") == "reviewed"
        and human_annotation.get("shot_type")
    ):
        reviewed_shot_type = str(human_annotation.get("shot_type")).strip()

    detected_shot_type = str(data.get("final_shot", "") or "").strip()
    resolved_shot_type = reviewed_shot_type or detected_shot_type

    if not analysis_valid:
        shot_type_display = "<font color='red'><b>N/A</b></font>"
    elif resolved_shot_type and resolved_shot_type.lower() not in ("unknown", "not provided", "n/a"):
        shot_type_display = (
            f"<font color='green'><b>{clean_pdf_text(resolved_shot_type)}</b></font>"
        )
    else:
        shot_type_display = "<font color='red'><b>Not provided</b></font>"

    footwork_data = final_report.get("footwork", {})
    footwork_stance = (
        footwork_data.get("stance", {})
        if isinstance(footwork_data, dict)
        else {}
    )

    batting_stance = (
        data.get("batting_stance")
        or footwork_stance.get("batting_stance")
        or "Unknown"
    )

    technique_score_display = (
        f"{float(technique_score):.1f} / 100"
        if analysis_valid and is_finite_number(technique_score)
        else "N/A"
    )

    score_display = (
        f"{float(final_score):.1f} / 100"
        if analysis_valid and is_finite_number(final_score)
        else "N/A"
    )

    impact_frame = (
        impact_events[0].get("impact_frame", "N/A") if has_impact else "None"
    )

    fps_val = float(source_info.get("fps", 0)) if is_finite_number(source_info.get("fps")) else 0.0

    exec_data = [
        [
            Paragraph("<b>Overall Joint Score:</b>", body_style),
            Paragraph(f"<b>{score_display}</b>", bold_body_style),
            Paragraph("<b>Shot Phase:</b>", body_style),
            Paragraph(clean_pdf_text(data.get("final_phase", "N/A")), body_style),
        ],
        [
            Paragraph("<b>Shot Status:</b>", body_style),
            Paragraph(
                (
                    "<font color='green'><b>Shot Played</b></font>"
                    if shot_status_lower == "shot played"
                    else "<font color='red'><b>Shot Not Played</b></font>"
                ),
                body_style,
            ),
            Paragraph("<b>Bat-Ball Impact:</b>", body_style),
            Paragraph(
                (
                    f"<font color='green'><b>YES (Frame {clean_pdf_text(impact_frame)})</b></font>"
                    if has_impact
                    else "<font color='red'>NO</font>"
                ),
                body_style,
            ),
        ],
        [
            Paragraph("<b>Frames Analyzed:</b>", body_style),
            Paragraph(
                f"{data.get('frames_processed', 0)} / {source_info.get('total_frames', 0)}",
                body_style,
            ),
            Paragraph("<b>Video Resolution:</b>", body_style),
            Paragraph(
                clean_pdf_text(
                    f"{source_info.get('width', 0)} × {source_info.get('height', 0)} @ {fps_val:.1f} FPS"
                ),
                body_style,
            ),
        ],
        [
            Paragraph("<b>Technique Score:</b>", body_style),
            Paragraph(technique_score_display, bold_body_style),
            Paragraph("<b>Batting Stance:</b>", body_style),
            Paragraph(clean_pdf_text(batting_stance), body_style),
        ],
        [
            Paragraph("<b>Shot Type:</b>", body_style),
            Paragraph(shot_type_display, body_style),
            Paragraph("", body_style),
            Paragraph("", body_style),
        ],
    ]

    exec_table = Table(exec_data, colWidths=[130, 140, 130, 140])
    exec_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
                ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(exec_table)
    story.append(Spacer(1, 6))

    if not analysis_valid:
        story.append(
            Paragraph(
                "<b>PERFORMANCE VALIDITY: INVALID</b><br/>"
                "No confirmed bat-ball impact is available in this report. "
                "Overall and joint performance scores are not valid. "
                "Any posture, movement, or technique observations below "
                "are diagnostic only.",
                body_style,
            )
        )
        story.append(Spacer(1, 5))

    if validity_mismatch:
        story.append(
            Paragraph(
                "<b>Data consistency warning:</b> The exported validity "
                "status and confirmed impact-event records do not agree. "
                "The report is treated as invalid.",
                body_style,
            )
        )
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 10))

    # ============================================================
    # SECTION 2: BIOMECHANICAL JOINT ANALYSIS
    # ============================================================
    story.append(Paragraph("2. Biomechanical Joint Analysis", h2_style))

    angles = final_report.get("angles", {}) or {}
    scores = final_report.get("scores", {}) or {}

    header_style = ParagraphStyle(
        "TableHeader",
        parent=bold_body_style,
        textColor=colors.white,
    )

    joint_table_data = [
        [
            Paragraph("<b>Joint Name</b>", header_style),
            Paragraph("<b>Measured Angle</b>", header_style),
            Paragraph("<b>Ideal Range</b>", header_style),
            Paragraph("<b>Joint Score</b>", header_style),
            Paragraph("<b>Status</b>", header_style),
        ]
    ]

    if angles:
        for joint, val in angles.items():
            j_score = scores.get(joint, "N/A")
            ideal_range = get_effective_joint_range(joint, batting_stance)

            ideal_range_display = (
                f"{ideal_range[0]}° – {ideal_range[1]}°"
                if analysis_valid and ideal_range is not None
                else "N/A"
            )

            if not analysis_valid:
                score_display_joint = "N/A"
                status_text = "N/A"
            elif is_finite_number(j_score):
                score_num = float(j_score)
                score_display_joint = f"{score_num:.1f}"
                status_text = (
                    "<font color='green'>Optimal</font>"
                    if score_num >= 80
                    else (
                        "<font color='orange'>Acceptable</font>"
                        if score_num >= 50
                        else "<font color='red'>Needs Work</font>"
                    )
                )
            else:
                score_display_joint = "N/A"
                status_text = "N/A"

            angle_str = f"{float(val):.1f}°" if is_finite_number(val) else str(val)

            joint_table_data.append(
                [
                    Paragraph(
                        clean_pdf_text(joint.replace("_", " ").title()),
                        body_style,
                    ),
                    Paragraph(clean_pdf_text(angle_str), body_style),
                    Paragraph(ideal_range_display, body_style),
                    Paragraph(score_display_joint, body_style),
                    Paragraph(status_text, body_style),
                ]
            )
    else:
        joint_table_data.append(
            [
                Paragraph("No joint metrics recorded.", body_style),
                Paragraph("-", body_style),
                Paragraph("-", body_style),
                Paragraph("-", body_style),
                Paragraph("-", body_style),
            ]
        )

    joint_table = Table(
        joint_table_data,
        colWidths=[125, 90, 100, 85, 140],
    )
    joint_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(joint_table)
    story.append(Spacer(1, 5))

    story.append(Paragraph("<b>Scoring methodology:</b>", bold_body_style))

    scoring_explanation = [
        "A joint receives 100 points when its angle is "
        "within the applicable ideal range.",
        "Outside the range, 2 points are deducted for "
        "each degree of distance from the nearest range boundary. "
        "The minimum joint score is 0.",
        "Overall Joint Score is the average of the available "
        "joint scores.",
        "Technique Score is the average joint score minus "
        "5 points for each actual technique mistake, with "
        "a minimum of 0.",
    ]

    for item in scoring_explanation:
        story.append(Paragraph(f"• {item}", footwork_feedback_style))

    story.append(
        Paragraph(
            "<i>Left-handed batting uses mirrored left/right "
            "joint reference ranges. Only available measured "
            "joints contribute to the average.</i>",
            body_style,
        )
    )

    story.append(Spacer(1, 10))

    # ============================================================
    # SECTION 3: TECHNIQUE & COACHING OBSERVATIONS
    # ============================================================
    story.append(Paragraph("3. Technique & Coaching Observations", h2_style))

    mistakes = final_report.get("mistakes", []) or []

    if not analysis_valid:
        story.append(
            Paragraph(
                "<b>Diagnostic-only observations:</b> "
                "These are frame-level observations and must not be "
                "interpreted as confirmed batting-performance flaws.",
                body_style,
            )
        )
        story.append(Spacer(1, 5))

    if mistakes:
        for mistake in mistakes:
            clean_flaw = clean_pdf_text(mistake)
            is_positive = "excellent" in str(mistake).lower()

            if is_positive:
                story.append(
                    Paragraph(
                        f"• <font color='green'><b>Positive Observation:</b></font> {clean_flaw}",
                        body_style,
                    )
                )
            else:
                observation_label = (
                    "Diagnostic Observation:"
                    if not analysis_valid
                    else "Flaw Detected:"
                )

                story.append(
                    Paragraph(
                        f"• <b>{observation_label}</b> {clean_flaw}",
                        body_style,
                    )
                )

                if analysis_valid:
                    advice = clean_pdf_text(get_correction_advice(mistake))
                    story.append(
                        Paragraph(
                            f"<b>Practice suggestion:</b> {advice}",
                            footwork_feedback_style,
                        )
                    )

            story.append(Spacer(1, 3))
    else:
        no_error_message = (
            "No major errors flagged in the available diagnostic frames."
            if not analysis_valid
            else "No major posture or technique errors flagged."
        )
        story.append(Paragraph(f"• {no_error_message}", body_style))

    story.append(Spacer(1, 10))

    # ============================================================
    # SECTION 4: FOOTWORK & KINEMATICS SUMMARY
    # ============================================================
    story.append(Paragraph("4. Footwork & Kinematics", h2_style))
    footwork = final_report.get("footwork", {})

    if isinstance(footwork, dict):
        fw_score = footwork.get("footwork_score", 0)
        clean_score = (
            clean_pdf_text(f"{fw_score} / 100")
            if analysis_valid
            else "N/A"
        )
        story.append(
            Paragraph(f"<b>Footwork Stability Score:</b> {clean_score}", body_style)
        )
        story.append(Spacer(1, 5))

        story.append(
            Paragraph("<b>Footwork scoring methodology:</b>", bold_body_style)
        )

        footwork_method = [
            "Head position: 20 points when the nose is within "
            "0.06 normalized units of the hip-centre X position.",
            "Front knee: 20 points when the measured angle is "
            "at or below 170°.",
            "Forward step: 40 points for a forward-step ratio "
            "of at least 0.08, or 20 points for a ratio of "
            "at least 0.04.",
            "Back-foot movement: 20 points when recent ankle movement "
            "is classified as either noticeable or low movement.",
        ]

        for item in footwork_method:
            story.append(Paragraph(f"• {item}", footwork_feedback_style))

        story.append(
            Paragraph(
                "<b>Total:</b> Head position + front knee + "
                "forward step + back-foot movement (maximum 100).",
                body_style,
            )
        )

        story.append(
            Paragraph(
                "<i>Hip-shift and screen-relative hip-centre "
                "movement are diagnostic only and do not contribute "
                "to the footwork score.</i>",
                body_style,
            )
        )

        story.append(Spacer(1, 5))
        feedback = footwork.get("feedback", []) or []

        for f in feedback:
            story.append(
                Paragraph(f"- {clean_pdf_text(f)}", footwork_feedback_style)
            )
    elif isinstance(footwork, list) and footwork:
        for item in footwork:
            story.append(Paragraph(f"• {clean_pdf_text(item)}", body_style))

    # Footer
    story.append(Spacer(1, 14))
    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=BORDER_COLOR,
            spaceBefore=0,
            spaceAfter=8,
        )
    )
    story.append(
        Paragraph(
            "<i>Generated automatically by AI Cricket Performance Analyzer Pipeline</i>",
            ParagraphStyle(
                "Footer",
                parent=body_style,
                fontSize=8,
                textColor=colors.gray,
            ),
        )
    )

    doc.build(story)
    return output_pdf_path