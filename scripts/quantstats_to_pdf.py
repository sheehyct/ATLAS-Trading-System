#!/usr/bin/env python3
"""
QuantStats HTML to PDF Converter with Risk Level Headers
=========================================================

Converts QuantStats-generated HTML tearsheets to PDF with proper SVG rendering
and optional risk level banners.

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    python quantstats_to_pdf.py report.html                    # Basic conversion
    python quantstats_to_pdf.py report.html --risk "2%"        # With risk header
    python quantstats_to_pdf.py report.html -o custom_name.pdf # Custom output name

Author: Generated for ATLAS Trading System
"""

import asyncio
import argparse
import os
import sys
import tempfile
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Error: playwright not installed. Run:")
    print("  pip install playwright")
    print("  playwright install chromium")
    sys.exit(1)


def prepare_html(input_path: str, risk_level: str = None) -> str:
    """
    Prepare HTML file by fixing broken JS and optionally adding risk header.
    Returns path to temporary prepared file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Fix broken QuantStats onload="save()"
    html = html.replace('onload="save()"', '')
    
    # Add risk level banner if specified
    if risk_level:
        risk_banner = f'''
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 12px 20px;
            margin-bottom: 20px;
            border-radius: 6px;
            border-left: 4px solid #e94560;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 13px; font-weight: 500;">ATLAS STRAT Options Backtest</div>
            <div style="
                background: #e94560;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
                letter-spacing: 0.5px;
            ">RISK LEVEL: {risk_level}</div>
        </div>
        '''
        html = html.replace(
            '<div class="container">',
            f'<div class="container">{risk_banner}'
        )
    
    # Write to temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.html')
    with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return temp_path


async def convert_to_pdf(html_path: str, pdf_path: str, scale: float = 0.85):
    """Convert HTML to PDF using Playwright/Chromium."""
    
    abs_html_path = os.path.abspath(html_path)
    file_url = f"file://{abs_html_path}"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Load page and wait for SVGs to render
        await page.goto(file_url, wait_until="networkidle")
        await page.wait_for_timeout(2000)
        
        # Generate PDF
        await page.pdf(
            path=pdf_path,
            format="Letter",
            print_background=True,
            margin={
                "top": "0.4in",
                "bottom": "0.4in", 
                "left": "0.4in",
                "right": "0.4in"
            },
            scale=scale
        )
        
        await browser.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert QuantStats HTML reports to PDF with SVG support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s report.html
  %(prog)s report.html --risk "2%%"
  %(prog)s report.html --risk "5%%" -o my_report.pdf
  %(prog)s *.html --risk "10%%"  # Batch convert
        """
    )
    parser.add_argument('files', nargs='+', help='HTML file(s) to convert')
    parser.add_argument('--risk', '-r', help='Risk level to display (e.g., "2%%", "5%%")')
    parser.add_argument('--output', '-o', help='Output PDF path (single file only)')
    parser.add_argument('--scale', type=float, default=0.85, help='PDF scale (default: 0.85)')
    
    args = parser.parse_args()
    
    if args.output and len(args.files) > 1:
        print("Error: --output can only be used with a single input file")
        sys.exit(1)
    
    for input_file in args.files:
        if not os.path.exists(input_file):
            print(f"Error: File not found: {input_file}")
            continue
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            stem = Path(input_file).stem
            output_path = f"{stem}.pdf"
        
        print(f"Converting: {input_file} -> {output_path}")
        
        # Prepare HTML
        temp_html = prepare_html(input_file, args.risk)
        
        try:
            # Convert to PDF
            asyncio.run(convert_to_pdf(temp_html, output_path, args.scale))
            print(f"  [OK] Success: {output_path}")
        finally:
            # Clean up temp file
            os.unlink(temp_html)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
