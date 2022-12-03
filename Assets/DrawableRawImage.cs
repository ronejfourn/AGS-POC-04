using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class DrawableRawImage : MonoBehaviour, IDragHandler, IPointerDownHandler, IPointerUpHandler
{
    public Camera drawCamera;
    public float width = 0.2f;
    public Color color = Color.white, clearColor = Color.black;
    LineRenderer m_LineRenderer;
    Vector2 m_PrevPos;

    void Start()
    {
        RectTransform rc = GetComponent<RectTransform>();
        rc.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, Screen.height);
        rc.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical  , Screen.height);

        RenderTexture rt = new RenderTexture((int)rc.rect.width, (int)rc.rect.height, 24);
        drawCamera.targetTexture = rt;
        RawImage ri = GetComponent<RawImage>();
        ri.texture = rt;

        m_LineRenderer = gameObject.AddComponent<LineRenderer>();
        m_LineRenderer.numCapVertices = 5;
        m_LineRenderer.positionCount  = 0;
        m_LineRenderer.material = new Material(Shader.Find("Sprites/Default"));

        ClearAll();
    }

    public void OnPointerDown(PointerEventData pd)
    {
        m_LineRenderer.startWidth = width;
        m_LineRenderer.endWidth   = width;
        m_LineRenderer.startColor = color;
        m_LineRenderer.endColor   = color;

        m_PrevPos = Camera.main.ScreenToWorldPoint(pd.position);
        m_LineRenderer.positionCount = 2;
        m_LineRenderer.SetPosition(0, m_PrevPos);
        m_LineRenderer.SetPosition(1, m_PrevPos);
    }

    public void OnPointerUp(PointerEventData pd)
    {
        m_LineRenderer.positionCount = 0;
    }

    public void OnDrag(PointerEventData pd)
    {
        Vector2 tempPos = Camera.main.ScreenToWorldPoint(pd.position);
        m_LineRenderer.SetPosition(0, m_PrevPos);
        m_LineRenderer.SetPosition(1, tempPos);
        m_PrevPos = tempPos;
    }

    public void ClearAll()
    {
        RenderTexture.active = drawCamera.targetTexture;
        GL.Clear(true, true, clearColor);
        RenderTexture.active = null;
    }

    public void SaveAsPNG()
    {
        RenderTexture rt = drawCamera.targetTexture;
        Texture2D t2D = new Texture2D(rt.width, rt.height);
        RenderTexture.active = rt;
        t2D.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        t2D.Apply();
        RenderTexture.active = null;
        byte[] bytes = t2D.EncodeToPNG();
        string time  = System.DateTime.Now.ToString("yyyyMMddHHmmss");
        string fpath = Application.persistentDataPath + "/SavedImage-" + time + ".png";
        System.IO.File.WriteAllBytes(fpath, bytes);
    }
}
