"""
Email service for sending profile creation notifications.

This module handles sending email notifications when a profile is created.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
from loguru import logger
import re


def extract_email_from_contact(contact: str) -> Optional[str]:
    """
    Extract email address from contact string.
    
    Args:
        contact: Contact string (can be email or phone)
        
    Returns:
        Email address if found, None otherwise
    """
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    # Check if contact is already an email
    if email_pattern.match(contact.strip()):
        return contact.strip()
    
    # Try to extract email from string
    email_match = re.search(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', contact)
    if email_match:
        return email_match.group()
    
    return None


def format_missing_person_email_html(metadata: Dict[str, Any], case_id: str, image_url: Optional[str] = None) -> str:
    """
    Format HTML email content for missing person profile.
    
    Args:
        metadata: Missing person metadata dictionary
        case_id: Case ID
        image_url: Optional image URL
        
    Returns:
        HTML formatted email content
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #4CAF50;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px 5px 0 0;
            }}
            .content {{
                background-color: #f9f9f9;
                padding: 20px;
                border: 1px solid #ddd;
            }}
            .info-section {{
                margin-bottom: 20px;
            }}
            .info-row {{
                margin: 10px 0;
                padding: 10px;
                background-color: white;
                border-left: 3px solid #4CAF50;
            }}
            .label {{
                font-weight: bold;
                color: #555;
            }}
            .value {{
                color: #333;
                margin-left: 10px;
            }}
            .footer {{
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                color: #777;
                font-size: 12px;
            }}
            .image-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>✅ Hồ Sơ Người Mất Tích Đã Được Tạo Thành Công</h1>
        </div>
        <div class="content">
            <p>Xin chào,</p>
            <p>Hồ sơ người mất tích của bạn đã được tạo thành công trong hệ thống. Dưới đây là thông tin chi tiết:</p>
            
            <div class="info-section">
                <div class="info-row">
                    <span class="label">Mã Hồ Sơ (Case ID):</span>
                    <span class="value">{case_id}</span>
                </div>
                <div class="info-row">
                    <span class="label">Họ và Tên:</span>
                    <span class="value">{metadata.get('name', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="label">Tuổi Khi Mất Tích:</span>
                    <span class="value">{metadata.get('age_at_disappearance', 'N/A')} tuổi</span>
                </div>
                <div class="info-row">
                    <span class="label">Năm Mất Tích:</span>
                    <span class="value">{metadata.get('year_disappeared', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="label">Giới Tính:</span>
                    <span class="value">{metadata.get('gender', 'N/A').upper()}</span>
                </div>
                <div class="info-row">
                    <span class="label">Địa Điểm Cuối Cùng:</span>
                    <span class="value">{metadata.get('location_last_seen', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="label">Thông Tin Liên Hệ:</span>
                    <span class="value">{metadata.get('contact', 'N/A')}</span>
                </div>
                {f'<div class="info-row"><span class="label">Chiều Cao:</span><span class="value">{metadata.get("height_cm")} cm</span></div>' if metadata.get('height_cm') else ''}
                {f'<div class="info-row"><span class="label">Đặc Điểm Nhận Dạng:</span><span class="value">{", ".join(metadata.get("birthmarks", []))}</span></div>' if metadata.get('birthmarks') else ''}
                {f'<div class="info-row"><span class="label">Thông Tin Bổ Sung:</span><span class="value">{metadata.get("additional_info")}</span></div>' if metadata.get('additional_info') else ''}
            </div>
            
            {f'<div class="image-container"><img src="{image_url}" alt="Ảnh người mất tích" /></div>' if image_url else ''}
            
            <p><strong>Lưu ý:</strong> Hệ thống đã tự động tìm kiếm các trường hợp khớp trong cơ sở dữ liệu. Bạn sẽ được thông báo nếu có kết quả khớp.</p>
        </div>
        <div class="footer">
            <p>Email này được gửi tự động từ hệ thống Tìm Kiếm Người Mất Tích</p>
            <p>Vui lòng không trả lời email này.</p>
        </div>
    </body>
    </html>
    """
    return html


def format_found_person_email_html(metadata: Dict[str, Any], found_id: str, image_url: Optional[str] = None) -> str:
    """
    Format HTML email content for found person profile.
    
    Args:
        metadata: Found person metadata dictionary
        found_id: Found ID
        image_url: Optional image URL
        
    Returns:
        HTML formatted email content
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #2196F3;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px 5px 0 0;
            }}
            .content {{
                background-color: #f9f9f9;
                padding: 20px;
                border: 1px solid #ddd;
            }}
            .info-section {{
                margin-bottom: 20px;
            }}
            .info-row {{
                margin: 10px 0;
                padding: 10px;
                background-color: white;
                border-left: 3px solid #2196F3;
            }}
            .label {{
                font-weight: bold;
                color: #555;
            }}
            .value {{
                color: #333;
                margin-left: 10px;
            }}
            .footer {{
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                color: #777;
                font-size: 12px;
            }}
            .image-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .image-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>✅ Hồ Sơ Người Được Tìm Thấy Đã Được Tạo Thành Công</h1>
        </div>
        <div class="content">
            <p>Xin chào,</p>
            <p>Hồ sơ người được tìm thấy của bạn đã được tạo thành công trong hệ thống. Dưới đây là thông tin chi tiết:</p>
            
            <div class="info-section">
                <div class="info-row">
                    <span class="label">Mã Hồ Sơ (Found ID):</span>
                    <span class="value">{found_id}</span>
                </div>
                {f'<div class="info-row"><span class="label">Tên:</span><span class="value">{metadata.get("name")}</span></div>' if metadata.get('name') else ''}
                <div class="info-row">
                    <span class="label">Tuổi Ước Tính Hiện Tại:</span>
                    <span class="value">{metadata.get('current_age_estimate', 'N/A')} tuổi</span>
                </div>
                <div class="info-row">
                    <span class="label">Giới Tính:</span>
                    <span class="value">{metadata.get('gender', 'N/A').upper()}</span>
                </div>
                <div class="info-row">
                    <span class="label">Vị Trí Hiện Tại:</span>
                    <span class="value">{metadata.get('current_location', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="label">Thông Tin Liên Hệ Người Tìm Thấy:</span>
                    <span class="value">{metadata.get('finder_contact', 'N/A')}</span>
                </div>
                {f'<div class="info-row"><span class="label">Dấu Hiệu Nhận Dạng:</span><span class="value">{", ".join(metadata.get("visible_marks", []))}</span></div>' if metadata.get('visible_marks') else ''}
                {f'<div class="info-row"><span class="label">Tình Trạng Hiện Tại:</span><span class="value">{metadata.get("current_condition")}</span></div>' if metadata.get('current_condition') else ''}
                {f'<div class="info-row"><span class="label">Thông Tin Bổ Sung:</span><span class="value">{metadata.get("additional_info")}</span></div>' if metadata.get('additional_info') else ''}
            </div>
            
            {f'<div class="image-container"><img src="{image_url}" alt="Ảnh người được tìm thấy" /></div>' if image_url else ''}
            
            <p><strong>Lưu ý:</strong> Hệ thống đã tự động tìm kiếm các trường hợp khớp trong cơ sở dữ liệu. Bạn sẽ được thông báo nếu có kết quả khớp.</p>
        </div>
        <div class="footer">
            <p>Email này được gửi tự động từ hệ thống Tìm Kiếm Người Mất Tích</p>
            <p>Vui lòng không trả lời email này.</p>
        </div>
    </body>
    </html>
    """
    return html


def send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str,
    to_email: str,
    subject: str,
    html_content: str,
    use_tls: bool = True
) -> bool:
    """
    Send email using SMTP.
    
    Args:
        smtp_host: SMTP server host
        smtp_port: SMTP server port
        smtp_user: SMTP username
        smtp_password: SMTP password
        from_email: Sender email address
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML email content
        use_tls: Whether to use TLS
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        
        # Create HTML part
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)
        
        # Connect to SMTP server
        if use_tls:
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port)
        
        # Login and send
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")
        return False


def send_missing_person_profile_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str,
    contact: str,
    metadata: Dict[str, Any],
    case_id: str,
    image_url: Optional[str] = None,
    use_tls: bool = True
) -> bool:
    """
    Send email notification for missing person profile creation.
    
    Args:
        smtp_host: SMTP server host
        smtp_port: SMTP server port
        smtp_user: SMTP username
        smtp_password: SMTP password
        from_email: Sender email address
        contact: Contact information (email or phone)
        metadata: Missing person metadata
        case_id: Case ID
        image_url: Optional image URL
        use_tls: Whether to use TLS
        
    Returns:
        True if email sent successfully, False otherwise
    """
    # Extract email from contact
    recipient_email = extract_email_from_contact(contact)
    
    if not recipient_email:
        logger.warning(f"No valid email found in contact: {contact}")
        return False
    
    # Format email content
    html_content = format_missing_person_email_html(metadata, case_id, image_url)
    subject = f"Hồ Sơ Người Mất Tích Đã Được Tạo - {case_id}"
    
    # Send email
    return send_email(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        from_email=from_email,
        to_email=recipient_email,
        subject=subject,
        html_content=html_content,
        use_tls=use_tls
    )


def send_found_person_profile_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str,
    finder_contact: str,
    metadata: Dict[str, Any],
    found_id: str,
    image_url: Optional[str] = None,
    use_tls: bool = True
) -> bool:
    """
    Send email notification for found person profile creation.
    
    Args:
        smtp_host: SMTP server host
        smtp_port: SMTP server port
        smtp_user: SMTP username
        smtp_password: SMTP password
        from_email: Sender email address
        finder_contact: Finder contact information (email or phone)
        metadata: Found person metadata
        found_id: Found ID
        image_url: Optional image URL
        use_tls: Whether to use TLS
        
    Returns:
        True if email sent successfully, False otherwise
    """
    # Extract email from contact
    recipient_email = extract_email_from_contact(finder_contact)
    
    if not recipient_email:
        logger.warning(f"No valid email found in finder_contact: {finder_contact}")
        return False
    
    # Format email content
    html_content = format_found_person_email_html(metadata, found_id, image_url)
    subject = f"Hồ Sơ Người Được Tìm Thấy Đã Được Tạo - {found_id}"
    
    # Send email
    return send_email(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        from_email=from_email,
        to_email=recipient_email,
        subject=subject,
        html_content=html_content,
        use_tls=use_tls
    )

