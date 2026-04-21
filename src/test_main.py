import functions_framework
import json


@functions_framework.http
def test_func(request):
    try:
        from main import get_all_users, get_drive_service, CONFIG

        drive = get_drive_service()
        pub = (
            drive.files()
            .get(fileId=CONFIG["PUBLIC_DRIVE_ID"], supportsAllDrives=True)
            .execute()
        )
        users = get_all_users()
        return json.dumps({"drive": pub.get("name"), "users": len(users)}), 200
    except Exception as e:
        return json.dumps({"error": str(e)}), 500
