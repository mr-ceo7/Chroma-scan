import os
import numpy as np
import matplotlib.pyplot as plt

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# ═══════════════════════════════════════════════════════════
#  DIAGRAM GENERATION
# ═══════════════════════════════════════════════════════════

def generate_diagrams():
    print("Generating diagrams...")
    
    # 1. Beer-Lambert Absorbance Curves
    fig, ax1 = plt.subplots(figsize=(6, 3.2))
    conc = np.linspace(0, 10, 100)
    trans = np.exp(-0.35 * conc)
    absorb = -np.log10(trans)
    
    ax1.set_xlabel('Concentration (c)', fontweight='bold', color='#1a202c')
    ax1.set_ylabel('Transmittance (I/I0)', color='#ef4444', fontweight='bold')
    ax1.plot(conc, trans, color='#ef4444', linewidth=2.5, label='Transmittance')
    ax1.tick_params(axis='y', labelcolor='#ef4444')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Absorbance (A)', color='#3b82f6', fontweight='bold')
    ax2.plot(conc, absorb, color='#3b82f6', linewidth=2.5, label='Absorbance')
    ax2.tick_params(axis='y', labelcolor='#3b82f6')
    
    plt.title('Absorbance & Transmittance vs. Concentration', fontsize=12, fontweight='bold', color='#1e3a8a')
    fig.tight_layout()
    plt.savefig('diagram_beer_lambert.png', dpi=300)
    plt.close()
    
    # 2. System Block Diagram
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.axis('off')
    
    # Draw boxes for ESP8266 and Arduino Nano
    esp = plt.Rectangle((0.08, 0.42), 0.38, 0.44, fill=True, color='#eff6ff', ec='#3b82f6', lw=2)
    nano = plt.Rectangle((0.54, 0.42), 0.38, 0.44, fill=True, color='#f0fdf4', ec='#22c55e', lw=2)
    
    ax.add_patch(esp)
    ax.add_patch(nano)
    
    # Texts
    ax.text(0.27, 0.64, 'ESP8266 Master MCU\n(WiFi, Web Server,\nOLED GUI Driver,\nState Machine, AI NN)', ha='center', va='center', fontweight='bold', color='#1e3a8a', fontsize=9)
    ax.text(0.73, 0.64, 'Arduino Nano Slave\n(Precise Photodiode ADC,\nRGB Excitation Switching,\nUART streaming,\nBeep Melodies)', ha='center', va='center', fontweight='bold', color='#14532d', fontsize=9)
    
    # Communication arrow
    ax.annotate('UART Serial\n(115200 bps)', xy=(0.54, 0.64), xytext=(0.46, 0.64),
                arrowprops=dict(arrowstyle="<->", color='#4b5563', lw=1.5),
                ha='center', va='center', fontsize=7.5, color='#4b5563', fontweight='bold')
    
    # External components
    oled = plt.Rectangle((0.08, 0.1), 0.17, 0.2, fill=True, color='#f9fafb', ec='#6b7280', lw=1.2)
    btns = plt.Rectangle((0.29, 0.1), 0.17, 0.2, fill=True, color='#f9fafb', ec='#6b7280', lw=1.2)
    leds = plt.Rectangle((0.54, 0.1), 0.17, 0.2, fill=True, color='#f9fafb', ec='#6b7280', lw=1.2)
    sensor = plt.Rectangle((0.75, 0.1), 0.17, 0.2, fill=True, color='#f9fafb', ec='#6b7280', lw=1.2)
    
    ax.add_patch(oled)
    ax.add_patch(btns)
    ax.add_patch(leds)
    ax.add_patch(sensor)
    
    ax.text(0.165, 0.2, 'OLED Display\n(I2C SSD1306)', ha='center', va='center', fontsize=7.5)
    ax.text(0.375, 0.2, '4x Buttons\n(UP/DN/OK/BK)', ha='center', va='center', fontsize=7.5)
    ax.text(0.625, 0.2, 'RGB + White\nLight Source', ha='center', va='center', fontsize=7.5)
    ax.text(0.835, 0.2, 'Photodiode\nDetector', ha='center', va='center', fontsize=7.5)
    
    # Connections
    ax.annotate('', xy=(0.165, 0.42), xytext=(0.165, 0.3), arrowprops=dict(arrowstyle="<-", color='#4b5563', lw=1))
    ax.annotate('', xy=(0.375, 0.42), xytext=(0.375, 0.3), arrowprops=dict(arrowstyle="->", color='#4b5563', lw=1))
    ax.annotate('', xy=(0.625, 0.42), xytext=(0.625, 0.3), arrowprops=dict(arrowstyle="<-", color='#4b5563', lw=1))
    ax.annotate('', xy=(0.835, 0.42), xytext=(0.835, 0.3), arrowprops=dict(arrowstyle="->", color='#4b5563', lw=1))
    
    plt.title('Chroma-Scan System Hardware Block Diagram', fontsize=12, fontweight='bold', color='#1e3a8a')
    plt.tight_layout()
    plt.savefig('diagram_system_architecture.png', dpi=300)
    plt.close()
    
    # 3. MLP Neural Network Topology
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.axis('off')
    
    inputs = ['Absorbance R', 'Absorbance G', 'Absorbance B', 'Ambient Light']
    hidden = [f'H{i}' for i in range(8)]
    outputs = ['Concentration']
    
    input_y = np.linspace(0.15, 0.85, len(inputs))
    hidden_y = np.linspace(0.08, 0.92, len(hidden))
    output_y = [0.5]
    
    # Connections
    for iy in input_y:
        for hy in hidden_y:
            ax.plot([0.16, 0.48], [iy, hy], color='#d1d5db', alpha=0.45, linewidth=0.7)
            
    for hy in hidden_y:
        for oy in output_y:
            ax.plot([0.48, 0.82], [hy, oy], color='#93c5fd', alpha=0.6, linewidth=1.0)
            
    # Nodes
    for i, y in enumerate(input_y):
        ax.plot(0.16, y, 'o', color='#3b82f6', markersize=12, zorder=3)
        ax.text(0.02, y, inputs[i], ha='left', va='center', fontweight='bold', fontsize=8.5, color='#1e3b8a')
        
    for j, y in enumerate(hidden_y):
        ax.plot(0.48, y, 'o', color='#818cf8', markersize=10, zorder=3)
        
    ax.text(0.48, 0.96, 'Hidden Layer (8 Neurons)', ha='center', va='bottom', fontsize=8, color='#4338ca', fontweight='bold')
        
    for k, y in enumerate(output_y):
        ax.plot(0.82, y, 'o', color='#10b981', markersize=15, zorder=3)
        ax.text(0.98, y, outputs[k], ha='right', va='center', fontweight='bold', fontsize=9, color='#065f46')
        
    plt.title('Edge AI MLP Feedforward Neural Network', fontsize=12, fontweight='bold', color='#1e3a8a')
    plt.tight_layout()
    plt.savefig('diagram_mlp_network.png', dpi=300)
    plt.close()
    
    print("Diagrams generated successfully!")

# ═══════════════════════════════════════════════════════════
#  PDF GENERATION & NUMBERED CANVAS
# ═══════════════════════════════════════════════════════════

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        if self._pageNumber == 1:
            self.saveState()
            self.setFont("Helvetica-Bold", 10)
            self.setFillColor(colors.HexColor("#1e3a8a"))
            self.drawCentredString(306, 45, "GALVANIY TECHNOLOGIES")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#475569"))
            self.drawCentredString(306, 32, "All Rights Reserved © 2026")
            self.restoreState()
            return
            
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1e3a8a"))
        
        # Header
        self.drawString(54, 752, "CHROMA-SCAN SPECTROPHOTOMETER")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(558, 752, "Technical & Operations Manual")
        
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 745, 558, 745)
        
        # Footer
        self.line(54, 52, 558, 52)
        self.drawString(54, 40, "GALVANIY TECHNOLOGIES | YSK 2026 Project 6")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        
        self.restoreState()

def build_pdf(filename="Chroma_Scan_Technical_Manual.pdf"):
    print(f"Building PDF: {filename}...")
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor("#1e3a8a"),
        alignment=0, # Left-aligned
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#475569"),
        spaceAfter=30
    )
    
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#1e293b")
    )
    
    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=22,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2563eb"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=0,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f8fafc"),
        borderColor=colors.HexColor("#e2e8f0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=8,
        spaceAfter=8
    )

    story = []

    # ═══════════════════════════════════════════════════════════
    #  COVER PAGE
    # ═══════════════════════════════════════════════════════════
    
    story.append(Spacer(1, 60))
    # Elegant top line decoration
    d_table = Table([[""]], colWidths=[504])
    d_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 4, colors.HexColor("#1e3a8a")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0)
    ]))
    story.append(d_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("CHROMA-SCAN<br/>SPECTROPHOTOMETER", title_style))
    story.append(Paragraph("Scientific Principles, Embedded System Architecture, Edge AI Modeling, and Complete Operations Manual", subtitle_style))
    
    story.append(Spacer(1, 15))
    story.append(Image("chromascan_device.png", width=220, height=220))
    story.append(Spacer(1, 25))
    
    # Metadata Block
    meta_text = """
    <b>Author / Lead Architect:</b> mr-ceo7 (Git)<br/>
    <b>Organization:</b> Galvaniy Technologies<br/>
    <b>Project Title:</b> YSK 2026 Project 6<br/>
    <b>Classification:</b> Open Source Technical Documentation<br/>
    <b>System Version:</b> v1.0 (Hardware: NodeMCU v2 / ESP8266 + Arduino Nano)<br/>
    <b>Date:</b> June 2026
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    #  CHAPTER 1: THEORETICAL PRINCIPLES
    # ═══════════════════════════════════════════════════════════
    
    story.append(Paragraph("1. Analytical Theory & Scientific Background", h1_style))
    
    theory_p1 = """
    Spectrophotometry is a quantitative method used in analytical chemistry to measure how much chemical substance absorbs light. 
    It leverages the relationship between concentration, path length, and absorption through the <b>Beer-Lambert Law</b>:
    """
    story.append(Paragraph(theory_p1, body_style))
    
    formula_style = ParagraphStyle(
        'FormulaStyle',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=11,
        alignment=1, # Center
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=8
    )
    story.append(Paragraph("A = &epsilon; &middot; b &middot; c = -log<sub>10</sub>( T ) = -log<sub>10</sub>( I / I<sub>0</sub> )", formula_style))
    
    theory_p2 = """
    Where:<br/>
    &bull; <b>A</b> is the Absorbance (dimensionless).<br/>
    &bull; <b>&epsilon;</b> is the molar attenuation coefficient (L &middot; mol<sup>-1</sup> &middot; cm<sup>-1</sup>).<br/>
    &bull; <b>b</b> is the path length of the cuvette (typically 1.0 cm).<br/>
    &bull; <b>c</b> is the molar concentration of the analyte (mol/L or mg/L).<br/>
    &bull; <b>T</b> is the Transmittance, representing the ratio of transmitted intensity (<b>I</b>) to the baseline light intensity (<b>I<sub>0</sub></b>).
    """
    story.append(Paragraph(theory_p2, body_style))
    
    story.append(Spacer(1, 10))
    story.append(Image("diagram_beer_lambert.png", width=340, height=180))
    story.append(Spacer(1, 5))
    story.append(Paragraph("<i>Figure 1.1: Graphical representation showing Transmittance decaying exponentially and Absorbance growing linearly as Concentration increases.</i>", ParagraphStyle('Cap', parent=body_style, fontName='Helvetica-Oblique', fontSize=8, alignment=1)))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Multi-Wavelength Sensing (RGB Spectroscopy)", h2_style))
    theory_p3 = """
    Standard single-wavelength colorimeters use filter wheels or single LEDs matching the wavelength of maximum absorbance (&lambda;<sub>max</sub>). 
    The Chroma-Scan utilizes three discrete narrow-band emission excitation wavelengths: <b>Red (625nm)</b>, <b>Green (525nm)</b>, and <b>Blue (465nm)</b>. 
    By measuring transmittance across three channels, the device can automatically select the <b>best wavelength channel</b> matching the complementary color of the analyte:
    """
    story.append(Paragraph(theory_p3, body_style))
    
    story.append(Paragraph("&bull; <b>Yellow-colored solutions</b> absorb blue light (465nm).", bullet_style))
    story.append(Paragraph("&bull; <b>Red/Magenta solutions</b> absorb green light (525nm).", bullet_style))
    story.append(Paragraph("&bull; <b>Blue/Cyan solutions</b> absorb red light (625nm).", bullet_style))
    
    theory_p4 = """
    <br/>
    <b>Beyond Beer-Lambert (Why Edge AI is Used):</b><br/>
    In physical setups, chemical compounds often deviate from the linear Beer-Lambert law due to chemical interactions, non-linear photodiode sensor responses, scattering, and ambient light leaks. 
    A traditional linear regression calibration curve will fail or saturates at higher concentrations. 
    By deploying an <b>Artificial Neural Network (MLP)</b>, the Chroma-Scan fits a non-linear multivariate function that models inter-wavelength combinations and ambient compensation, significantly expanding the device's operational range and accuracy.
    """
    story.append(Paragraph(theory_p4, body_style))
    
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    #  CHAPTER 2: HARDWARE ARCHITECTURE
    # ═══════════════════════════════════════════════════════════
    
    story.append(Paragraph("2. System Hardware & Embedded Interfacing", h1_style))
    
    hw_p1 = """
    The device is designed as a split-architecture system consisting of two controllers communicating over UART. This ensures that time-critical sensor operations do not conflict with heavy networking, web serving, and OLED UI processing.
    """
    story.append(Paragraph(hw_p1, body_style))
    
    story.append(Spacer(1, 10))
    story.append(Image("diagram_system_architecture.png", width=350, height=210))
    story.append(Spacer(1, 5))
    story.append(Paragraph("<i>Figure 2.1: Master/Slave split microcontroller architecture of the Chroma-Scan device.</i>", ParagraphStyle('Cap2', parent=body_style, fontName='Helvetica-Oblique', fontSize=8, alignment=1)))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("ESP8266 Master Controller", h2_style))
    hw_p2 = """
    The NodeMCU v2 (ESP8266) acts as the Master Controller. It manages the central user interface, local state machine, graphics on the I2C OLED screen, handles debounced tactile buttons, hosts the WiFi Access Point Portal (`192.168.4.1`), handles HTTP API endpoints, parses JSON parameters, and computes the Edge AI neural network feedforward algorithm.
    """
    story.append(Paragraph(hw_p2, body_style))
    
    story.append(Paragraph("Arduino Nano ADC Slave co-processor", h2_style))
    hw_p3 = """
    The Arduino Nano acts as the dedicated measurement unit. Upon receiving a command over UART from the master:
    1. It turns off all light sources to read raw <b>ambient background level</b>.
    2. Sequentially turns on <b>Red, Green, Blue, and White</b> LEDs.
    3. Samples light intensity on a high-precision photodiode connected to an analog pin, averaging multiple samples over time to reject 50/60Hz grid flicker.
    4. Streams raw numbers back to the ESP8266 and triggers buzzer indicators.
    """
    story.append(Paragraph(hw_p3, body_style))
    
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    #  CHAPTER 3: SOFTWARE & EMBEDDED SAFETY
    # ═══════════════════════════════════════════════════════════
    
    story.append(Paragraph("3. Software Safeguards & Runtime Watchdogs", h1_style))
    
    sw_p1 = """
    Operating web servers, REST JSON parsers, file systems, and float-heavy neural network matrices on a resource-constrained microcontroller like the ESP8266 (with ~80KB RAM) can easily lead to memory leakage, heap fragmentation, and crash loops. 
    To prevent these issues, the master firmware implements the following safety constraints:
    """
    story.append(Paragraph(sw_p1, body_style))
    
    story.append(Paragraph("<b>1. RAM Heap Watchdog:</b>", h2_style))
    sw_p2 = """
    A background system task periodically checks available RAM. If the free heap falls below <b>6KB</b>, the system outputs warning diagnostics to Serial, halts processing, and performs an automatic software restart to clear memory fragmentation before a memory allocation crash occurs.
    """
    story.append(Paragraph(sw_p2, body_style))
    
    story.append(Paragraph("<b>2. Write Interception and Filesystem Safeguards:</b>", h2_style))
    sw_p3 = """
    During model parameter uploads:
    - The server verifies that there is at least <b>20KB</b> of storage free on LittleFS before starting.
    - Each block chunk is written and verified. If a write fails or storage is exhausted, a global flag `uploadError` is set, canceling the operation and avoiding parsing incomplete, corrupt JSON configurations.
    """
    story.append(Paragraph(sw_p3, body_style))
    
    story.append(Paragraph("<b>3. CSV Data File Cap:</b>", h2_style))
    sw_p4 = """
    Data points gathered in the calibration studio are appended to a `/standards.csv` file. 
    If the file grows to **50KB** (representing approximately 1,200 points), saving is disabled. This prevents the filesystem from filling up and locking out web server file writes.
    """
    story.append(Paragraph(sw_p4, body_style))
    
    story.append(Spacer(1, 10))
    story.append(Image("diagram_mlp_network.png", width=340, height=190))
    story.append(Spacer(1, 5))
    story.append(Paragraph("<i>Figure 3.1: Network topology of the embedded feedforward MLP model.</i>", ParagraphStyle('Cap3', parent=body_style, fontName='Helvetica-Oblique', fontSize=8, alignment=1)))
    
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    #  CHAPTER 4: STEP-BY-STEP OPERATION
    # ═══════════════════════════════════════════════════════════
    
    story.append(Paragraph("4. Step-by-Step Operations Manual", h1_style))
    
    # --- SECTION 4.1 ---
    story.append(Paragraph("4.1 Startup and Menu Navigation", h2_style))
    op_p1 = """
    Upon power-on, the device performs hardware self-checks and displays welcoming sequence screens, followed by <i>'Chroma-Scan Ready. Press OK...'</i>. 
    Pressing <b>OK</b> opens the main carousel menu. Use <b>UP</b> and <b>DOWN</b> buttons to scroll through the apps, and press <b>OK</b> to select.
    """
    story.append(Paragraph(op_p1, body_style))
    
    # --- SECTION 4.2 ---
    story.append(Paragraph("4.2 Blank Solvent Calibration", h2_style))
    op_p2 = """
    <b>Critical Pre-requisite:</b> You must calibrate the blank reference before any prediction scan. This sets the baseline light intensities (I<sub>0</sub>).
    1. Select the <b>Calibration App</b> (Option 1).
    2. Place a cuvette with distilled water (or clear reference solvent) in the sample chamber.
    3. Close the lid and press <b>OK</b> to scan.
    4. The device plays a laser animation and registers raw RGB values.
    5. **Error Check:** If raw readings are below 50 (e.g. lid open, opaque cuvette, or blocked sensor), the screen displays a <b>[!] CALIB ERROR</b> warning. Clear the path and re-calibrate.
    """
    story.append(Paragraph(op_p2, body_style))
    
    # --- SECTION 4.3 ---
    story.append(Paragraph("4.3 Dataset Studio & Model Upload", h2_style))
    op_p3 = """
    To collect training data:
    1. Select **AI Dataset Studio** (Option 2). The OLED screen will show WiFi information (SSID: `Chroma-Scan-AP`, IP: `192.168.4.1`).
    2. Connect your smartphone/computer to the Access Point and open `http://192.168.4.1/`.
    3. Choose a concentration entry mode (Presets on OLED, or Manual entry on web).
    4. Insert calibration standards (e.g. 1.0, 2.0, 5.0 mg/L) one by one and click **Scan** on the web.
    5. The web page draws the real-time calibration curve and updates the standards table.
    6. Click **Download CSV** to download the `standards.csv` file.
    7. Start your local training server (`python3 app.py`), upload your CSV, train your model, and download `model_params.json`.
    8. Back on the device portal, drag and drop `model_params.json` and click **Deploy Model**. The OLED plays a downloading animation and updates weights in LittleFS.
    """
    story.append(Paragraph(op_p3, body_style))
    
    # --- SECTION 4.4 ---
    story.append(Paragraph("4.4 Running Edge AI Inference", h2_style))
    op_p4 = """
    Once calibrated and trained:
    1. Open **Edge AI Prediction** (Option 3).
    2. Select the inference model (e.g., `Chroma-MLP` or `Beer-Lambert fallback`). 
       - **Error Check:** If you select Custom AI but no parameters are uploaded, the screen displays a <b>[!] NO AI MODEL</b> warning.
    3. Insert your unknown concentration sample and press **Scan**.
    4. The screen plays the scanning animation followed by the neural network feedforward calculation animation.
    5. Read the estimated concentration and absorbance scores on the multi-page results screen.
    """
    story.append(Paragraph(op_p4, body_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF build complete!")

if __name__ == "__main__":
    generate_diagrams()
    build_pdf()
