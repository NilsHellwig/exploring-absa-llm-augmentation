import datetime

def format_seconds_to_time_string(total_seconds):
    time_duration = datetime.timedelta(seconds=total_seconds)
    hours, remainder = divmod(time_duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_format = ""
    
    if hours > 0:
        time_format += f"{hours}h "
    
    if minutes > 0:
        time_format += f"{minutes}m "
    
    time_format += f"{seconds}s"

    return time_format