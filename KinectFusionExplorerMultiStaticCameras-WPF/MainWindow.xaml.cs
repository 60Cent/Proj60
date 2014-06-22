//------------------------------------------------------------------------------
// <copyright file="MainWindow.xaml.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace Microsoft.Samples.Kinect.KinectFusionExplorer
{
    using System;
    using System.Collections.Generic;
    using System.ComponentModel;
    using System.Globalization;
    using System.IO;
    using System.Text;
    using System.Threading.Tasks;
    using System.Windows;
    using System.Windows.Controls;
    using System.Windows.Data;
    using System.Windows.Media;
    using System.Windows.Media.Imaging;
    using System.Windows.Media.Media3D;
    using System.Windows.Threading;
    using Microsoft.Kinect;
    using Microsoft.Kinect.Toolkit;
    using Microsoft.Kinect.Toolkit.Fusion;
    using Wpf3DTools;
    /// <summary>
    /// Interaction logic for the <see cref="MainWindow"/> class
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged, IDisposable
    {
        #region Constants

        /// <summary>
        /// The minimum count of frames used for reconstructing
        /// </summary>
        public const int MinReconstructionFrameInCount = 1;

        /// <summary>
        /// The maximum count of frames used for reconstructing
        /// </summary>
        public const int MaxReconstructionFrameInCount = 200;

        /// <summary>
        /// Maximum number of sensors to support for reconstruction
        /// </summary>
        private const int MaxSensors = 3;

        /// <summary>
        /// Event interval for FPS timer
        /// </summary>
        private const int FpsInterval = 5;

        /// <summary>
        /// Event interval for status bar timer
        /// </summary>
        private const int StatusBarInterval = 1;

        private bool tracking = false;
        Matrix4 CameraChangeMatrix4;

        /// <summary>
        /// The reconstruction volume processor type. This parameter sets whether AMP or CPU processing
        /// is used. Note that CPU processing will likely be too slow for real-time processing.
        /// </summary>
        private const ReconstructionProcessor ProcessorType = ReconstructionProcessor.Amp;

        /// <summary>
        /// The zero-based device index to choose for reconstruction processing if the 
        /// ReconstructionProcessor AMP options are selected.
        /// Here we automatically choose a device to use for processing by passing -1, 
        /// </summary>
        private const int DeviceToUse = -1;

        /// <summary>
        /// Image size of depth frame to use
        /// </summary>
        private const DepthImageFormat DepthFormat = DepthImageFormat.Resolution640x480Fps30;

        /// <summary>
        /// Image size of color frame to use
        /// </summary>
        private const ColorImageFormat ColorFormat = ColorImageFormat.RgbResolution640x480Fps30;

        /// <summary>
        /// The count of frames to wait as a lead in before we start reconstructing (e.g. for when we turn the IR Laser on or off)
        /// </summary>
        private const int PerCameraReconstructionFrameLeadInCount = 0;// 30;

        /// <summary>
        /// WPF3D Origin coordinate cross axis size in m
        /// </summary>
        private const float OriginCoordinateCrossAxisSize = 0.1f;

        /// <summary>
        /// Volume Cube and Origin coordinate cross axis 3D graphics line thickness in screen pixels
        /// </summary>
        private const int LineThickness = 2;

        /// <summary>
        /// The far plane distance for the camera frustum 3d graphics in m
        /// </summary>
        private const float CameraFrustum3DGraphicsFarPlaneDistance = 1.0f;

        #endregion

        #region Fields

        /// <summary>
        /// Volume Cube 3D graphics line color
        /// Green, partly transparent
        /// </summary>
        private static System.Windows.Media.Color volumeCubeLineColor = System.Windows.Media.Color.FromArgb(200, 0, 200, 0);

        /// <summary>
        /// Track whether Dispose has been called
        /// </summary>
        private bool disposed;

        /// <summary>
        /// Saving mesh flag
        /// </summary>
        private bool savingMesh;

        /// <summary>
        /// Image width of depth frame
        /// </summary>
        private int depthWidth = 0;

        /// <summary>
        /// Image height of depth frame
        /// </summary>
        private int depthHeight = 0;

        /// <summary>
        /// The counter for frames that have been processed in fps counter
        /// </summary>
        private int processedFrameCountForFps = 0;

        /// <summary>
        /// Timer to count FPS
        /// </summary>
        private DispatcherTimer fpsTimer;

        /// <summary>
        /// Timer stamp of last computation of FPS
        /// </summary>
        private DateTime lastFPSTimestamp;

        /// <summary>
        /// Timer used for ensuring status bar message will be displayed at least one second
        /// </summary>
        private DispatcherTimer statusBarTimer;

        /// <summary>
        /// Timer stamp of last update of status message
        /// </summary>
        private DateTime lastStatusTimestamp;

        /// <summary>
        /// Represent last saving sensor setting status message
        /// </summary>
        private string lastSensorSettingStatus = string.Empty;

        /// <summary>
        /// Saving status message
        /// </summary>
        private Queue<string> statusMessageQueue = new Queue<string>();

        /// <summary>
        /// Kinect sensors chooser objects
        /// </summary>
        private List<KinectSensorChooser> sensorChoosers = new List<KinectSensorChooser>();

        /// <summary>
        /// Active Kinect sensors
        /// </summary>
        private List<ReconstructionSensor> sensors = new List<ReconstructionSensor>();

        /// <summary>
        /// The Kinect Fusion volume
        /// </summary>
        private ColorReconstruction volume;

        /// <summary>
        /// Bitmap contains depth float frame data for rendering
        /// </summary>
        private WriteableBitmap depthFloatFrameBitmap;

        /// <summary>
        /// Bitmap contains shaded surface frame data for rendering
        /// </summary>
        private WriteableBitmap shadedSurfaceFrameBitmap;

        /// <summary>
        /// Bitmap that will hold color information
        /// </summary>
        private WriteableBitmap colorFrameBitmap;

        /// <summary>
        /// Pixel buffer of depth float frame with pixel data in float format
        /// </summary>
        private float[] depthFloatFrameDepthPixels;

        /// <summary>
        /// Pixel buffer of depth float frame with pixel data in 32bit color
        /// </summary>
        private int[] depthFloatFramePixelsArgb;

        /// <summary>
        /// Pixels buffer of shaded surface frame in 32bit color
        /// </summary>
        private int[] shadedSurfaceFramePixelsArgb;

        /// <summary>
        /// The actual transformation between the world and volume coordinate system we create
        /// </summary>
        private Matrix4 worldToVolumeTransform;

        /// <summary>
        /// To display shaded surface normals frame instead of shaded surface frame
        /// </summary>
        private bool displayNormals;

        /// <summary>
        /// Pause or resume image integration
        /// </summary>
        private bool pauseIntegration;

        /// <summary>
        /// Image integration weight
        /// </summary>
        private short integrationWeight = 1;//FusionDepthProcessor.DefaultIntegrationWeight;

        /// <summary>
        /// The reconstruction volume voxel density in voxels per meter (vpm)
        /// 1000mm / 256vpm = ~3.9mm/voxel
        /// </summary>
        private float voxelsPerMeter = 128.0f;

        /// <summary>
        /// The reconstruction volume voxel resolution in the X axis
        /// At a setting of 256vpm the volume is 512 / 256 = 2m wide
        /// </summary>
        private int voxelsX = 128;//512;

        /// <summary>
        /// The reconstruction volume voxel resolution in the Y axis
        /// At a setting of 256vpm the volume is 384 / 256 = 1.5m high
        /// </summary>
        private int voxelsY = 128;//384;

        /// <summary>
        /// The reconstruction volume voxel resolution in the Z axis
        /// At a setting of 256vpm the volume is 512 / 256 = 2m deep
        /// </summary>
        private int voxelsZ = 128;//512;

        FusionPointCloudImageFrame currentPCL;
        FusionPointCloudImageFrame oldPCL;

        /// <summary>
        /// The reconstruction is integrating frames
        /// </summary>
        private bool reconstructing = false;

        /// <summary>
        /// The reconstruction camera currently integrating frames
        /// </summary>
        private int reconstructingCamera = 0;

        /// <summary>
        /// The count of frames to integrate in reconstruction
        /// </summary>
        private int perCameraReconstructionFrameCount = 10;

        /// <summary>
        /// The camera tab selected
        /// </summary>
        private int cameraTabSelected = 0;

        /// <summary>
        /// The first frame flag, so we can do initialization processing
        /// </summary>
        private bool firstFrame = true;

        /// <summary>
        /// The counter for frames that have been processed
        /// </summary>
        private int processedFrameCount = 0;

        /// <summary>
        /// The counter for lead in frames before we start capturing (e.g. for when we turn the IR Laser on or off)
        /// </summary>
        private int processedFrameLeadInCount = 0;

        /// <summary>
        /// Load and save settings
        /// </summary>
        private bool loadSavePerKinectSettings = true;

        /// <summary>
        /// Lock for reconstruction
        /// </summary>
        private object reconstructionLock = new object();

        /// <summary>
        /// The volume cube 3D graphical representation
        /// </summary>
        private ScreenSpaceLines3D volumeCube;

        /// <summary>
        /// The volume cube 3D graphical representation
        /// </summary>
        private ScreenSpaceLines3D volumeCubeAxisX;

        /// <summary>
        /// The volume cube 3D graphical representation
        /// </summary>
        private ScreenSpaceLines3D volumeCubeAxisY;

        /// <summary>
        /// The volume cube 3D graphical representation
        /// </summary>
        private ScreenSpaceLines3D volumeCubeAxisZ;

        /// <summary>
        /// The axis-aligned coordinate cross X axis
        /// </summary>
        private ScreenSpaceLines3D axisX;

        /// <summary>
        /// The axis-aligned coordinate cross Y axis
        /// </summary>
        private ScreenSpaceLines3D axisY;

        /// <summary>
        /// The axis-aligned coordinate cross Z axis
        /// </summary>
        private ScreenSpaceLines3D axisZ;

        /// <summary>
        /// Flag set true to use individual camera view during reconstruction
        /// </summary>
        private bool useCameraViewInReconstruction = false;

        /// <summary>
        /// Flag boolean set true to force the reconstruction visualization to be updated after graphics camera movements
        /// </summary>
        private bool viewChanged = true;

        /// <summary>
        /// Indicate whether the 3D view port has added the volume cube
        /// </summary>
        private bool haveAddedVolumeCube = false;

        /// <summary>
        /// Indicate whether the 3D view port has added the origin coordinate cross
        /// </summary>
        private bool haveAddedCoordinateCross = false;

        /// <summary>
        /// The virtual 3rd person camera view that can be controlled by the mouse
        /// </summary>
        private GraphicsCamera virtualCamera;

        /// <summary>
        /// The virtual 3rd person camera view that can be controlled by the mouse - start rotation
        /// </summary>
        private Quaternion virtualCameraStartRotation = Quaternion.Identity;

        /// <summary>
        /// The virtual 3rd person camera view that can be controlled by the mouse - start translation
        /// </summary>
        private Point3D virtualCameraStartTranslation = new Point3D(); // 0,0,0

        /// <summary>
        /// Whether to see the main user controlled graphics camera to the sensor poses on reset, or keep separate
        /// </summary>
        private bool resetToFirstValidCameraView = false;

        /// <summary>
        /// Flag set true if at some point color has been captured. 
        /// Used when writing .Ply mesh files to output vertex color.
        /// </summary>
        private bool colorCaptured;

        #endregion

        /// <summary>
        /// Initializes a new instance of the MainWindow class.
        /// </summary>
        public MainWindow()
        {
            this.InitializeComponent();
        }

        /// <summary>
        /// Finalizes an instance of the MainWindow class.
        /// This destructor will run only if the Dispose method does not get called.
        /// </summary>
        ~MainWindow()
        {
            this.Dispose(false);
        }

        #region Properties

        /// <summary>
        /// Called asynchronously to process a depth frame and reconstruct
        /// </summary>
        /// <param name="sensor">The reconstruction sensor.</param>
        private delegate void ReconstructFrameDelegate(ReconstructionSensor sensor);

        /// <summary>
        /// Called asynchronously to render reconstruction from sensor or graphics camera pose
        /// </summary>
        /// <param name="sensor">The reconstruction sensor.</param>
        private delegate void RenderFrameDelegate(ReconstructionSensor sensor);

        /// <summary>
        /// Property change event
        /// </summary>
        public event PropertyChangedEventHandler PropertyChanged;

        /// <summary>
        /// Gets or sets the count of frames to capture
        /// </summary>
        public double CaptureFrames
        {
            get
            {
                return (double)this.perCameraReconstructionFrameCount;
            }

            set
            {
                this.perCameraReconstructionFrameCount = (int)(value + 0.5);
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("CaptureFrames"));
                }
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to display surface normals
        /// </summary>
        public bool DisplayNormals
        {
            get
            {
                return this.displayNormals;
            }

            set
            {
                this.displayNormals = value;
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("DisplayNormals"));
                }
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to pause depth data integration
        /// </summary>
        public bool PauseIntegration
        {
            get
            {
                return this.pauseIntegration;
            }

            set
            {
                this.pauseIntegration = value;
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("PauseIntegration"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the maximum weight value to use in depth data integration
        /// </summary>
        public double IntegrationWeight
        {
            get
            {
                return (double)this.integrationWeight;
            }

            set
            {
                this.integrationWeight = (short)(value + 0.5);
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("IntegrationWeight"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the voxels per meter
        /// </summary>
        public double VoxelsPerMeter
        {
            get
            {
                return (double)this.voxelsPerMeter;
            }

            set
            {
                this.voxelsPerMeter = (float)value;
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("VoxelsPerMeter"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the  X-axis volume resolution
        /// </summary>
        public double VoxelsX
        {
            get
            {
                return (double)this.voxelsX;
            }

            set
            {
                this.voxelsX = (int)(value + 0.5);
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("VoxelsX"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the Y-axis volume resolution
        /// </summary>
        public double VoxelsY
        {
            get
            {
                return (double)this.voxelsY;
            }

            set
            {
                this.voxelsY = (int)(value + 0.5);
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("VoxelsY"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the Z-axis volume resolution
        /// </summary>
        public double VoxelsZ
        {
            get
            {
                return (double)this.voxelsZ;
            }

            set
            {
                this.voxelsZ = (int)(value + 0.5);
                if (null != this.PropertyChanged)
                {
                    this.PropertyChanged.Invoke(this, new PropertyChangedEventArgs("VoxelsZ"));
                }
            }
        }

        /// <summary>
        /// Gets or sets the point cloud frame calculated either from the Reconstruction virtualCamera point of view
        /// </summary>
        private FusionPointCloudImageFrame PointCloudFrame { get; set; }

        /// <summary>
        /// Gets or sets the shaded surface frame from shading point cloud from the Reconstruction virtualCamera point of view
        /// </summary>
        private FusionColorImageFrame ShadedSurfaceFrame { get; set; }

        /// <summary>
        /// Gets or sets the shaded surface normals frame from shading point cloud from the Reconstruction virtualCamera point of view
        /// </summary>
        private FusionColorImageFrame ShadedSurfaceNormalsFrame { get; set; }

        #endregion

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            this.Dispose(true);

            // This object will be cleaned up by the Dispose method.
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Frees associated memory and tidies up
        /// </summary>
        /// <param name="disposing">Whether the function was called from Dispose.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!this.disposed)
            {
                if (disposing)
                {
                    foreach (ReconstructionSensor sensor in this.sensors)
                    {
                        // Turn all Lasers off and dispose sensors
                        this.ChangeSensorEmitterState(sensor, true);

                        sensor.Dispose();
                    }

                    if (null != this.volume)
                    {
                        this.volume.Dispose();
                    }

                    this.RemoveVolumeCube3DGraphics();
                    this.DisposeVolumeCube3DGraphics();

                    this.RemoveAxisAlignedCoordinateCross3DGraphics();
                    this.DisposeAxisAlignedCoordinateCross3DGraphics();

                    if (null != this.virtualCamera)
                    {
                        this.virtualCamera.CameraTransformationChanged -= this.OnVirtualCameraTransformationChanged;
                        this.virtualCamera.Detach(this.shadedSurfaceImage); // stop getting mouse events from the image
                        this.virtualCamera.Dispose();
                    }
                }
            }

            this.disposed = true;
        }

        /// <summary>
        /// Render Fusion color frame to UI
        /// </summary>
        /// <param name="colorFrame">Fusion color frame</param>
        /// <param name="colorPixels">Pixel buffer for fusion color frame</param>
        /// <param name="bitmap">Bitmap contains color frame data for rendering</param>
        /// <param name="image">UI image component to render the color frame</param>
        private static void RenderColorImage(
            FusionColorImageFrame colorFrame, ref int[] colorPixels, ref WriteableBitmap bitmap, System.Windows.Controls.Image image)
        {
            if (null == colorFrame)
            {
                return;
            }

            if (null == colorPixels || colorFrame.PixelDataLength != colorPixels.Length)
            {
                // Create pixel array of correct format
                colorPixels = new int[colorFrame.PixelDataLength];
            }

            if (null == bitmap || colorFrame.Width != bitmap.Width || colorFrame.Height != bitmap.Height)
            {
                // Create bitmap of correct format
                bitmap = new WriteableBitmap(colorFrame.Width, colorFrame.Height, 96.0, 96.0, PixelFormats.Bgr32, null);

                // Set bitmap as source to UI image object
                image.Source = bitmap;
            }

            // Copy pixel data to pixel buffer
            colorFrame.CopyPixelDataTo(colorPixels);

            // Write pixels to bitmap
            bitmap.WritePixels(new Int32Rect(0, 0, colorFrame.Width, colorFrame.Height), colorPixels, bitmap.PixelWidth * sizeof(int), 0);
        }

        /// <summary>
        /// Render Fusion color frame to UI direct from byte buffer
        /// </summary>
        /// <param name="colorPixels">Pixel buffer for fusion color frame.</param>
        /// <param name="width">The width of the image.</param>
        /// <param name="height">The height of the image.</param>
        /// <param name="bitmap">Bitmap contains color frame data for rendering.</param>
        /// <param name="image">UI image component to render the color frame.</param>
        private static void RenderColorImage(
            byte[] colorPixels, int width, int height, ref WriteableBitmap bitmap, System.Windows.Controls.Image image)
        {
            if (null == colorPixels)
            {
                return;
            }

            if (null == bitmap || width != bitmap.Width || height != bitmap.Height)
            {
                // Create bitmap of correct format
                bitmap = new WriteableBitmap(width, height, 96.0, 96.0, PixelFormats.Bgr32, null);

                // Set bitmap as source to UI image object
                image.Source = bitmap;
            }

            // Write pixels to bitmap
            bitmap.WritePixels(
                new Int32Rect(0, 0, width, height),
                colorPixels,
                bitmap.PixelWidth * sizeof(byte) * 4, // rgba
                0);
        }

        /// <summary>
        /// Execute startup tasks
        /// </summary>
        /// <param name="sender">Object sending the event</param>
        /// <param name="e">Event arguments</param>
        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
            int deviceMemoryKB = 0;

            // Check to ensure suitable DirectX11 compatible hardware exists before initializing Kinect Fusion
            try
            {
                string deviceDescription = string.Empty;
                string deviceInstancePath = string.Empty;

                FusionDepthProcessor.GetDeviceInfo(
                    ProcessorType, DeviceToUse, out deviceDescription, out deviceInstancePath, out deviceMemoryKB);
            }
            catch (IndexOutOfRangeException)
            {
                // Thrown when index is out of range for processor type or there is no DirectX11 capable device installed.
                // As we set -1 (auto-select default) for the DeviceToUse above, this indicates that there is no DirectX11 
                // capable device. The options for users in this case are to either install a DirectX11 capable device 
                // (see documentation for recommended GPUs) or to switch to non-real-time CPU based reconstruction by 
                // changing ProcessorType to ReconstructionProcessor.Cpu
                this.statusBarText.Text = Properties.Resources.NoDirectX11CompatibleDeviceOrInvalidDeviceIndex;
                return;
            }
            catch (DllNotFoundException)
            {
                this.statusBarText.Text = Properties.Resources.MissingPrerequisite;
                return;
            }

            const int Mebi = 1024 * 1024;
            bool is32Bit = sizeof(int) == IntPtr.Size;

            // We now create both a color and depth volume, doubling the required memory, so we restrict
            // which resolution settings the user can choose when the graphics card is limited in memory.
            if (deviceMemoryKB <= 1 * Mebi)
            {
                // Disable 640 voxel resolution in all axes - cards with only 1GB cannot handle this
                VoxelsXSlider.Maximum = 512;
                VoxelsYSlider.Maximum = 512;
                VoxelsZSlider.Maximum = 512;

                // Also disable 512 voxel resolution in one arbitrary axis on 32bit machines
                if (is32Bit)
                {
                    VoxelsYSlider.Maximum = 384;
                }
            }
            else if (deviceMemoryKB <= 2 * Mebi)
            {
                // Disable 640 voxel resolution in one arbitrary axis on 32bit machines
                if (is32Bit)
                {
                    VoxelsYSlider.Maximum = 512;
                }
            }

            // Allocate temporary storage for virtualCamera viewpoint renderings
            Size imageSize = Helper.GetImageSize(DepthFormat);
            this.depthWidth = (int)imageSize.Width;
            this.depthHeight = (int)imageSize.Height;

            // Setup the graphics rendering

            // Create virtualCamera for non-Kinect viewpoint rendering
            // Default position translated along Z axis, looking back at origin
            this.virtualCameraStartTranslation = new Point3D(0, 0, this.voxelsZ / this.voxelsPerMeter);
            this.virtualCamera = new GraphicsCamera(
                this.virtualCameraStartTranslation, this.virtualCameraStartRotation, (float)Width / (float)Height);

            // Enable user control through the mouse over the reconstruction image
            this.virtualCamera.Attach(this.shadedSurfaceImage); // get mouse events from the image to update the camera
            this.virtualCamera.CameraTransformationChanged += this.OnVirtualCameraTransformationChanged;

            // Attach this camera to the viewport
            this.GraphicsViewport.Camera = this.virtualCamera.Camera;

            // Create axis-aligned coordinate cross 3D graphics at the WPF3D/reconstruction world origin
            // Red is the +X axis, Green is the +Y axis, Blue is the +Z axis in the WPF3D coordinate system
            // Note that the coordinate cross shows the WPF3D coordinate system (right hand, erect so +Y up and +X right, +Z out of screen), rather 
            // than the volume reconstruction coordinate system (right hand, rotated so +Y is down and +X is right, +Z into screen ).
            this.CreateAxisAlignedCoordinateCross3DGraphics(new Point3D(0, 0, 0), OriginCoordinateCrossAxisSize, LineThickness);
            this.AddAxisAlignedCoordinateCross3DGraphics();
             
            // Start first Kinect sensors chooser
            // Start this visible so we see the "connect sensor" message if none are plugged in 
            this.AddSensorChooser(new Thickness(10, 0, 0, 5), false);

            // Add callback which is called every time WPF renders
            System.Windows.Media.CompositionTarget.Rendering += this.CompositionTargetRendering;

            if (null == this.sensors || (null != this.sensors && 0 == this.sensors.Count))
            {
                this.ShowStatusMessage(Properties.Resources.NoReadyKinect);
            }

            // Start fps timer
            this.fpsTimer = new DispatcherTimer(DispatcherPriority.Send);
            this.fpsTimer.Interval = new TimeSpan(0, 0, FpsInterval);
            this.fpsTimer.Tick += this.FpsTimerTick;
            this.fpsTimer.Start();

            // Set last fps timestamp as now
            this.lastFPSTimestamp = DateTime.Now;

            // Start status bar timer
            this.statusBarTimer = new DispatcherTimer(DispatcherPriority.Send);
            this.statusBarTimer.Interval = new TimeSpan(0, 0, StatusBarInterval);
            this.statusBarTimer.Tick += this.StatusBarTimerTick;
            this.statusBarTimer.Start();

            this.lastStatusTimestamp = DateTime.Now;
        }

        /// <summary>
        /// Allocate the frame buffers used for rendering virtualCamera
        /// </summary>
        private void AllocateFrames()
        {
            // Allocate point cloud frame
            if (null == this.PointCloudFrame || this.depthWidth != this.PointCloudFrame.Width
                || this.depthHeight != this.PointCloudFrame.Height)
            {
                this.PointCloudFrame = new FusionPointCloudImageFrame(this.depthWidth, this.depthHeight);
            }

            // Allocate shaded surface frame
            if (null == this.ShadedSurfaceFrame || this.depthWidth != this.ShadedSurfaceFrame.Width
                || this.depthHeight != this.ShadedSurfaceFrame.Height)
            {
                this.ShadedSurfaceFrame = new FusionColorImageFrame(this.depthWidth, this.depthHeight);
            }

            // Allocate shaded surface normals frame
            if (null == this.ShadedSurfaceNormalsFrame || this.depthWidth != this.ShadedSurfaceNormalsFrame.Width
                || this.depthHeight != this.ShadedSurfaceNormalsFrame.Height)
            {
                this.ShadedSurfaceNormalsFrame = new FusionColorImageFrame(this.depthWidth, this.depthHeight);
            }
        }

        /// <summary>
        /// Called on each render of WPF (usually around 60Hz)
        /// </summary>
        /// <param name="sender">Object sending the event</param>
        /// <param name="e">Event arguments</param>
        private void CompositionTargetRendering(object sender, EventArgs e)
        {
            // If not reconstructing then the viewChanged flag is used so we only raycast the volume when something changes
            // When reconstructing we call RenderReconstruction manually for every integrated depth frame (see ReconstructDepthData)
            if (!this.reconstructing && this.viewChanged)
            {
                this.RenderReconstruction(null); // passing null renders from the virtualCamera pose
                this.viewChanged = false;
            }
        }

        /// <summary>
        /// Add Sensor Chooser objects to UI
        /// </summary>
        /// <param name="margin">The margin to set around the sensor chooser icon in the UI.</param>
        /// <param name="hide"> Whether to create the sensor chooser with a hidden icon.</param>
        private void AddSensorChooser(Thickness margin, bool hide)
        {
            this.sensorChoosers.Add(new KinectSensorChooser());
            KinectSensorChooser sensorChooser = this.sensorChoosers[this.sensorChoosers.Count - 1];

            this.sensorchooserStackPanel.Children.Add(new KinectSensorChooserUI());
            KinectSensorChooserUI ui =
                this.sensorchooserStackPanel.Children[this.sensorchooserStackPanel.Children.Count - 1] as KinectSensorChooserUI;
            if (ui != null)
            {
                ui.Margin = margin;
            }

            ui.KinectSensorChooser = sensorChooser;
            sensorChooser.KinectChanged += this.OnKinectSensorChanged;
            sensorChooser.Start();

            if (hide)
            {
                ui.Visibility = Visibility.Hidden;
            }
        }

        /// <summary>
        /// Remove Sensor Chooser objects
        /// </summary>
        /// <param name="deviceConnectionId">The device ID for the sensor chooser to remove.</param>
        /// <param name="hideUnpluggedSensorChooser">Set true to immediately hide the unplugged sensor chooser.</param>
        private void RemoveSensorChooser(string deviceConnectionId, bool hideUnpluggedSensorChooser)
        {
            // Hide unplugged sensor chooser
            if (hideUnpluggedSensorChooser)
            {
                int resultIdx = this.sensors.FindIndex(0, c => (c.Sensor.DeviceConnectionId == deviceConnectionId));
                if (0 <= resultIdx && resultIdx < this.sensorchooserStackPanel.Children.Count)
                {
                    this.sensorchooserStackPanel.Children[resultIdx].Visibility = Visibility.Hidden;
                }
            }

            // As long as this is not the only one, we remove the last spare, unused sensor chooser,
            // as we now have another unused one
            if (1 < this.sensorChoosers.Count && 1 < this.sensorchooserStackPanel.Children.Count)
            {
                this.sensorChoosers.RemoveAt(this.sensorChoosers.Count - 1);
                this.sensorchooserStackPanel.Children.RemoveAt(this.sensorchooserStackPanel.Children.Count - 1);
            }

            // Make sure if we have only one that it is forced to be visible so it will ask users to plug in a sensor.
            if (1 == this.sensorChoosers.Count && 1 == this.sensorchooserStackPanel.Children.Count)
            {
                this.sensorchooserStackPanel.Children[0].Visibility = Visibility.Visible;
            }
        }

        /// <summary>
        /// Execute shutdown tasks
        /// </summary>
        /// <param name="sender">Object sending the event</param>
        /// <param name="e">Event arguments</param>
        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            // Stop timer
            if (null != this.fpsTimer)
            {
                this.fpsTimer.Stop();
                this.fpsTimer.Tick -= this.FpsTimerTick;
            }

            if (null != this.statusBarTimer)
            {
                this.statusBarTimer.Stop();
                this.statusBarTimer.Tick -= this.StatusBarTimerTick;
            }

            // Unregister Kinect sensors chooser event
            foreach (KinectSensorChooser sensorChooser in this.sensorChoosers)
            {
                sensorChooser.KinectChanged -= this.OnKinectSensorChanged;
            }

            // Stop sensors
            foreach (ReconstructionSensor sensor in this.sensors)
            {
                this.UnsubscribeAndStopSensor(sensor);
            }
        }

        /// <summary>
        /// Handler function for Kinect changed event
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnKinectSensorChanged(object sender, KinectChangedEventArgs e)
        {
            // Check new Sensor's status
            if (null != e.NewSensor)
            {
                ReconstructionSensor result = null;
                result = this.sensors.Find(c => (c.Sensor.DeviceConnectionId == e.NewSensor.DeviceConnectionId));

                // new sensor
                if (null == result && KinectStatus.Connected == e.NewSensor.Status)
                {
                    // Make sensor chooser visible
                    KinectSensorChooserUI ui =
                        this.sensorchooserStackPanel.Children[this.sensorchooserStackPanel.Children.Count - 1] as KinectSensorChooserUI;
                    if (ui != null)
                    {
                        ui.Visibility = Visibility.Visible;
                    }

                    // Allocate frames and Create volume if at least one Kinect is attached
                    this.AllocateFrames();

                    if (null == this.volume)
                    {
                        if (this.RecreateReconstruction())
                        {
                            // Show introductory message
                            this.ShowStatusMessage(Properties.Resources.IntroductoryMessage);
                        }
                    }

                    this.sensors.Add(new ReconstructionSensor(e.NewSensor));

                    ReconstructionSensor sensor = this.sensors[this.sensors.Count - 1];

                    bool isSupportNearMode = this.StartDepthStream(sensor, DepthFormat);
                    this.StartColorStream(sensor, ColorFormat);

                    if (this.virtualCamera != null && sensor.ReconCamera != null)
                    {
                        if (this.resetToFirstValidCameraView && this.sensors.Count == 1)
                        {
                            // Set the initial raycast virtual camera view to the first camera view
                            this.virtualCamera.WorldToCameraMatrix3D = this.sensors[0].ReconCamera.WorldToCameraMatrix3D;
                        }

                        // Add the camera frustum graphics
                        sensor.ReconCamera.CreateFrustum3DGraphics(this.GraphicsViewport, this.depthWidth, this.depthHeight);
                        sensor.ReconCamera.AddFrustum3DGraphics();
                    }
                    else
                    {
                        this.sensors.Remove(sensor);
                        sensor.Dispose();

                        return;
                    }

                    this.AddSensorTabControl(sensor, isSupportNearMode);
                    sensor.RequireResetEvent += this.ResetReconstruction;

                    // These two have the same effect and just redraw the reconstruction
                    sensor.RequireRenderEvent += this.OnKinectSensorTransformationChanged;
                    sensor.SensorTransformationChanged += this.OnKinectSensorTransformationChanged;
                    sensor.AllSetCaptureColorEvent += this.OnCaptureColorChanged;

                    bool loadedSettings = false;
                    if (this.loadSavePerKinectSettings)
                    {
                        loadedSettings = sensor.LoadSettings();
                    }

                    if (loadedSettings)
                    {
                        this.ShowStatusMessage(
                            string.Format(CultureInfo.InvariantCulture, "Camera {0} : {1}", this.sensors.Count - 1, Properties.Resources.LoadSensorSettings));
                    }
                    else
                    {
                        this.ShowStatusMessage(
                            string.Format(CultureInfo.InvariantCulture, "Camera {0} : {1}", this.sensors.Count - 1, Properties.Resources.LoadDefaultSettings));
                    }

                    if (!this.loadSavePerKinectSettings || (this.loadSavePerKinectSettings && false == loadedSettings))
                    {
                        // Force an update of the camera transformation from the UI
                        sensor.UpdateCameraTransformation();
                    }

                    // Create new sensor chooser for new sensors
                    if (this.sensorChoosers.Count < MaxSensors)
                    {
                        // Start hidden so we don't see the "connect sensor" message 
                        this.AddSensorChooser(new Thickness(80, 0, 0, 5), true);
                    }
                }
            }
            else if (null != e.OldSensor)
            {
                // Sensor removed
                ReconstructionSensor result = null;
                result = this.sensors.Find(c => (c.Sensor.DeviceConnectionId == e.OldSensor.DeviceConnectionId));

                if (null != result && KinectStatus.Connected != e.OldSensor.Status)
                {
                    this.RemoveSensorTabControl(e.OldSensor.DeviceConnectionId);

                    // Removes the last spare, unused sensor chooser and then sets the sensor chooser for the
                    // camera we just unplugged  hidden so we don't see the "connect sensor" message 
                    this.RemoveSensorChooser(e.OldSensor.DeviceConnectionId, true);

                    this.UnsubscribeAndStopSensor(result);

                    this.sensors.Remove(result);

                    result.ReconCamera.DisposeFrustum3DGraphics();

                    result.Dispose();
                }
            }
        }

        /// <summary>
        /// Un-subscribe from events and stop sensor streams
        /// </summary>
        /// <param name="sensor">The sensor object to stop.</param>
        private void UnsubscribeAndStopSensor(ReconstructionSensor sensor)
        {
            if (null == sensor)
            {
                return;
            }

            sensor.DepthFrameReady -= this.OnDepthFrameReady;
            sensor.ColorFrameReady -= this.OnColorFrameReady;
            sensor.RequireResetEvent -= this.ResetReconstruction;
            sensor.RequireRenderEvent -= this.OnKinectSensorTransformationChanged;
            sensor.SensorTransformationChanged -= this.OnKinectSensorTransformationChanged;
            sensor.AllSetCaptureColorEvent -= this.OnCaptureColorChanged;
            sensor.StopDepthStream();
            sensor.StopColorStream();
        }

        /// <summary>
        /// Handler for FPS timer tick
        /// </summary>
        /// <param name="sender">Object sending the event</param>
        /// <param name="e">Event arguments</param>
        private void FpsTimerTick(object sender, EventArgs e)
        {
            if (!this.savingMesh)
            {
                if (null == this.sensors)
                {
                    // Show "No ready Kinect found!" on status bar
                    this.statusBarText.Text = Properties.Resources.NoReadyKinect;
                }
                else
                {
                    // Calculate time span from last calculation of FPS
                    double intervalSeconds = (DateTime.Now - this.lastFPSTimestamp).TotalSeconds;

                    // Calculate and show fps on status bar
                    this.statusBarText.Text = string.Format(
                        System.Globalization.CultureInfo.InvariantCulture,
                        Properties.Resources.Fps,
                        (double)this.processedFrameCountForFps / intervalSeconds);
                }
            }

            // Reset frame counter
            this.processedFrameCountForFps = 0;
            this.lastFPSTimestamp = DateTime.Now;
        }

        /// <summary>
        /// Handler for status bar timer tick
        /// </summary>
        /// <param name="sender">Object sending the event</param>
        /// <param name="e">Event arguments</param>
        private void StatusBarTimerTick(object sender, EventArgs e)
        {
            if (this.statusMessageQueue.Count > 0)
            {
                this.statusBarText.Text = this.statusMessageQueue.Dequeue();

                // Update the last timestamp of status message
                this.lastStatusTimestamp = DateTime.Now;
            }
        }

        /// <summary>
        /// Reset FPS timer and counter
        /// </summary>
        private void ResetFps()
        {
            // Restart fps timer
            if (null != this.fpsTimer)
            {
                this.fpsTimer.Stop();
                this.fpsTimer.Start();
            }

            // Reset frame counter
            this.processedFrameCountForFps = 0;
            this.lastFPSTimestamp = DateTime.Now;
        }

        /// <summary>
        /// Start depth stream at specific resolution
        /// </summary>
        /// <param name="sensor">The reconstruction sensor instance.</param>
        /// <param name="format">The resolution of image in depth stream.</param>
        /// <returns>Returns true if the sensor supports near mode.</returns>
        private bool StartDepthStream(ReconstructionSensor sensor, DepthImageFormat format)
        {
            if (sensor == null)
            {
                return true;
            }

            bool isSupportNearMode = true;
            try
            {
                // Enable depth stream, register event handler and start
                sensor.DepthFrameReady += this.OnDepthFrameReady;
                isSupportNearMode = sensor.StartDepthStream(format);
            }
            catch (IOException ex)
            {
                // Device is in use
                this.ShowStatusMessage(ex.Message);

                return isSupportNearMode;
            }
            catch (InvalidOperationException ex)
            {
                // Device is not valid, not supported or hardware feature unavailable
                this.ShowStatusMessage(ex.Message);

                return isSupportNearMode;
            }

            try
            {
                // Make sure Lasers are turned on
                sensor.Sensor.ForceInfraredEmitterOff = false;
            }
            catch (InvalidOperationException ex)
            {
                // Device is not valid, not supported or hardware feature unavailable
                // show an error message just this once
                this.ShowStatusMessage(ex.Message);
            }

            return isSupportNearMode;
        }

        /// <summary>
        /// Start color stream at specific resolution
        /// </summary>
        /// <param name="sensor">The reconstruction sensor instance.</param>
        /// <param name="format">The resolution of image in color stream.</param>
        private void StartColorStream(ReconstructionSensor sensor, ColorImageFormat format)
        {
            if (sensor == null)
            {
                return;
            }

            try
            {
                // Enable color stream, register event handler and start
                sensor.ColorFrameReady += this.OnColorFrameReady;
                sensor.StartColorStream(format);
            }
            catch (IOException ex)
            {
                // Device is in use
                this.ShowStatusMessage(ex.Message);

                return;
            }
            catch (InvalidOperationException ex)
            {
                // Device is not valid, not supported or hardware feature unavailable
                this.ShowStatusMessage(ex.Message);

                return;
            }
        }

        /// <summary>
        /// Event raised when the mouse updates the graphics camera transformation for the virtual camera
        /// Here we set the viewChanged flag to true, to cause a volume render when the WPF composite update event occurs
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnVirtualCameraTransformationChanged(object sender, EventArgs e)
        {
            this.viewChanged = true;
        }

        /// <summary>
        /// Event raised when the UI updates the Kinect camera transformations from the sliders
        /// Here we set the viewChanged flag to true, to cause a volume render when the WPF composite update event occurs
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnKinectSensorTransformationChanged(object sender, EventArgs e)
        {
            this.viewChanged = true;

            // Set the frustum graphics transform directly from this
            ReconstructionSensor sensor = sender as ReconstructionSensor;
            if (null != sensor)
            {
                sensor.ReconCamera.UpdateFrustumTransformMatrix3D(sensor.ReconCamera.CameraToWorldMatrix3D);
            }
        }

        /// <summary>
        /// Event raised when the UI updates the Kinect capture color value
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnCaptureColorChanged(object sender, EventArgs e)
        {
            if (sender is ReconstructionSensor)
            {
                bool newCaptureColor = (sender as ReconstructionSensor).CaptureColor;

                foreach (ReconstructionSensor sensor in this.sensors)
                {
                    sensor.UpdateCaptureColor(newCaptureColor);
                }
            }
        }

        /// <summary>
        /// Called when Reconstruct button pushed to start reconstruction process
        /// </summary>
        /// <returns>Returns true if no error on saving settings, false otherwise.</returns>
        private bool[] Reconstruct()
        {
            if (this.sensors.Count < 1 || this.reconstructing)
            {
                return null;
            }

            bool[] successfulSave = null;
            if (this.loadSavePerKinectSettings && this.sensors.Count > 0)
            {
                successfulSave = new bool[this.sensors.Count];

                for (int i = 0; i < this.sensors.Count; i++)
                {
                    successfulSave[i] = this.sensors[i].SaveSettings();
                }
            }

            // start the reconstruction process
            this.reconstructingCamera = 0;
            this.reconstructing = true;
            this.processedFrameCount = 0;
            this.processedFrameLeadInCount = 0;

            return successfulSave;
        }

        /// <summary>
        /// Called when we have iterated through all valid cameras, reset the reconstruction or recreated the reconstruction
        /// </summary>
        private void DoneReconstructing()
        {
            this.reconstructing = false;
            this.reconstructingCamera = 0;

            this.processedFrameCount = 0;
            this.processedFrameLeadInCount = 0;

            foreach (ReconstructionSensor s in this.sensors)
            {
                // Make sure all Lasers are turned back on after reconstructing
                this.ChangeSensorEmitterState(s, false);
            }

            // Force raycast of one frame
            this.viewChanged = true;
        }

        /// <summary>
        /// Go to the next camera index during reconstruction.
        /// </summary>
        /// <returns>Returns true if the next camera was found, or false if we reach the end of the cameras.</returns>
        private bool GoToNextCameraIndex()
        {
            // Go to next camera
            ++this.reconstructingCamera;
            this.processedFrameCount = 0;
            this.processedFrameLeadInCount = 0;

            if (this.reconstructingCamera >= this.sensors.Count)
            {
                // Done reconstructing all cameras
                this.DoneReconstructing();
                return true;
            }

            return false;
        }

        /// <summary>
        /// Add a sensor control tab to the tab collection.
        /// </summary>
        /// <param name="sensor">The sensor to add a control for.</param>
        /// <param name="isSupportNearMode">Indicate whether the sensor supports near mode.</param>
        private void AddSensorTabControl(ReconstructionSensor sensor, bool isSupportNearMode)
        {
            ItemCollection items = tabControl.Items;

            items.Add(new TabItem());
            int index = items.Count - 1;
            TabItem t = items[index] as TabItem;

            t.Header = "Camera " + index.ToString(CultureInfo.CurrentCulture); // sensor.UniqueKinectId could also be used
            t.Content = sensor.ReconSensorControl;

            if (!isSupportNearMode)
            {
                var sensorControl = t.Content as ReconstructionSensorControl;
                sensorControl.checkBoxNearMode.IsEnabled = false;
            }

            // Select this added tab index
            tabControl.SelectedIndex = index;
        }

        /// <summary>
        /// Remove a sensor control tab from the tab collection.
        /// </summary>
        /// <param name="deviceConnectionId">The device ID to remove.</param>
        private void RemoveSensorTabControl(string deviceConnectionId)
        {
            int resultIdx = this.sensors.FindIndex(0, c => (c.Sensor.DeviceConnectionId == deviceConnectionId));

            // Remove sensor control also
            ItemCollection items = tabControl.Items;

            try
            {
                items.RemoveAt(resultIdx);
            }
            catch (InvalidOperationException)
            {
                // Fail silently
            }
            catch (ArgumentOutOfRangeException)
            {
                // Fail silently
            }

            // re-order existing tab numbers
            for (int i = resultIdx; i < items.Count; ++i)
            {
                TabItem t = items[i] as TabItem;
                t.Header = "Camera " + i.ToString(CultureInfo.CurrentCulture); // sensor.UniqueKinectId could also be used
            }
        }

        /// <summary>
        /// Event handler for Kinect sensor's DepthFrameReady event
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnDepthFrameReady(object sender, EventArgs e)
        {
            // Open depth frame
            ReconstructionSensor sensor = sender as ReconstructionSensor;

            if (sensor == null)
            {
                return;
            }

            if (this.reconstructing)
            {
                try
                {
                    // Test to see if this is the sensor we want to integrate frames from
                    int idx = this.sensors.FindIndex(c => c.Sensor.DeviceConnectionId == sensor.Sensor.DeviceConnectionId);

                    if (idx == this.reconstructingCamera)
                    {
                        // Skip any cameras we don't want to use
                        if (this.sensors[idx].UseSensor == false)
                        {
                            if (this.GoToNextCameraIndex())
                            {
                                // out of cameras 
                                this.ShowStatusMessage(this.lastSensorSettingStatus + Properties.Resources.DoneReconstructing);
                                return;
                            }
                            else
                            {
                                return; // reconstructingCamera has been incremented and we will check in the next depth frame
                            }
                        }

                        if (this.processedFrameLeadInCount > PerCameraReconstructionFrameLeadInCount)
                        {
                            if (this.processedFrameCount >= this.perCameraReconstructionFrameCount && !this.tracking)
                            {
                                if (this.GoToNextCameraIndex())
                                {
                                    this.ShowStatusMessage(this.lastSensorSettingStatus + Properties.Resources.DoneReconstructing);
                                }

                                return;
                            }

                            // Run this asynchronously, but update the UI by calling an async update function from inside the ReconstructDepthData function
                            ReconstructFrameDelegate reconstructNewFrame = this.ReconstructDepthData;
                            reconstructNewFrame.BeginInvoke(sensor, null, null);

                            // Increase processed frame counters
                            ++this.processedFrameCount;
                            ++this.processedFrameCountForFps;
                        }
                        else
                        {
                            // Turn this IR Laser on, and the others off when reconstructing
                            // This reduces interference when cameras are pointed in similar directions
                            // The lead-in period gives the sensor time to stabilize the depth
                            if (this.processedFrameLeadInCount == 0)
                            {
                                for (int i = 0; i < this.sensors.Count; i++)
                                {
                                    if (i == idx)
                                    {
                                        // Turn on this Laser for reconstruction
                                        this.ChangeSensorEmitterState(this.sensors[i], false);
                                    }
                                    else
                                    {
                                        // Turn off this Laser after reconstruction completes
                                        this.ChangeSensorEmitterState(this.sensors[i], true);
                                    }
                                }
                            }

                            ++this.processedFrameLeadInCount;

                            this.statusBarText.Text = Properties.Resources.LeadInFrame
                                                      + this.processedFrameLeadInCount.ToString(CultureInfo.CurrentCulture);
                        }
                    }
                }
                catch (ArgumentNullException)
                {
                    // Fail silently
                }
            }
            else
            {
                try
                {
                    // Just display the depth image for the selected camera tab
                    int cameraIndex = this.cameraTabSelected;

                    if (this.reconstructing && this.useCameraViewInReconstruction)
                    {
                        // Display image for camera if we are reconstructing and want this instead
                        cameraIndex = this.reconstructingCamera;
                    }

                    if (this.sensors.FindIndex(c => c.Sensor.DeviceConnectionId == sensor.Sensor.DeviceConnectionId) == cameraIndex)
                    {
                        this.PreProcessDepthData(sensor);

                        ++this.processedFrameCountForFps;
                    }
                }
                catch (ArgumentNullException)
                {
                    // Fail silently
                }
            }
        }

        /// <summary>
        /// Event handler for Kinect sensor's ColorFrameReady event
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void OnColorFrameReady(object sender, EventArgs e)
        {
            // Open color frame
            ReconstructionSensor sensor = sender as ReconstructionSensor;

            if (sensor == null)
            {
                return;
            }

            // Display the color image only for the selected camera tab
            int cameraIndex = this.cameraTabSelected;

            if (this.reconstructing && this.useCameraViewInReconstruction)
            {
                // Display image for camera if we are reconstructing and want this instead
                cameraIndex = this.reconstructingCamera;
            }

            if (!sensor.MirrorDepth)
            {
                sensor.MirrorColorHorizontalInPlace();
            }

            if (this.sensors.FindIndex(c => c.Sensor.DeviceConnectionId == sensor.Sensor.DeviceConnectionId) == cameraIndex)
            {
                // Use dispatcher object to invoke PreProcessDepthData function to process
                RenderColorImage(sensor.ColorImagePixels, sensor.ColorWidth, sensor.ColorHeight, ref this.colorFrameBitmap, this.colorImage);
            }
        }

        /// <summary>
        /// Just convert to float and draw the depth frame
        /// </summary>
        /// <param name="sensor">The sensor where the depth frame originated.</param>
        private void PreProcessDepthData(ReconstructionSensor sensor)
        {
            if (sensor == null)
            {
                return;
            }

            // Check near mode
            //sensor.CheckNearMode();

            // Convert depth frame to depth float frame
            this.volume.DepthToDepthFloatFrame(
                sensor.DepthImagePixels, sensor.DepthFloatFrame, sensor.MinDepthClip, sensor.MaxDepthClip, sensor.MirrorDepth);

            // Run the UI update on the UI thread
            Dispatcher.BeginInvoke((Action)(() => this.DepthFrameComplete(sensor)));
        }

        /// <summary>
        /// Process the depth input
        /// </summary>
        /// <param name="sensor">The the sensor to use in reconstruction.</param>
        bool tracked = false;
        float accuracyOfTrack = 0;
        private void doDepthStuff(ReconstructionSensor sensor)
        {
            tracked = this.volume.ProcessFrame(
                sensor.DepthFloatFrame,
                7,
                this.integrationWeight,
                sensor.ReconCamera.WorldToCameraMatrix4);
            
            if (tracked)
            {
                this.CameraChangeMatrix4 = this.volume.GetCurrentWorldToCameraTransform();// sensor.ReconCamera.WorldToCameraMatrix4;
                this.integrationWeight = this.integrationWeight++;
                if (this.integrationWeight > 1000)
                {
                    this.integrationWeight = 1000;
                }
                if (Math.Abs(accuracyOfTrack) < 0.1)
                {
                    sensor.ReconCamera.UpdateFrustumTransformMatrix4(CameraChangeMatrix4);
                    short[] volumeBlock = new short[2097152];
                    
                    
                    this.volume.ExportVolumeBlock(0, 0, 0, 128, 128, 128, 1, volumeBlock);
                    int calcVolume = 0;
                    //for (int i = 0; i < 2097152; i++)
                    //{
                    //    if (volumeBlock[i] != 0x0000)
                    //    {
                    //        calcVolume++;
                    //    }

                    //}

                    this.VolumeLabel.Text = calcVolume.ToString();
                    //sensor.ReconSensorControl.AngleX = 180/Math.PI*Math.Atan2(CameraChangeMatrix4.M21, CameraChangeMatrix4.M21);

                    //sensor.ReconSensorControl.AngleY = 180 / Math.PI * Math.Atan2(CameraChangeMatrix4.M31,
                    //Math.Sqrt(
                    //(
                    //(CameraChangeMatrix4.M32) * (CameraChangeMatrix4.M32)
                    //) + (
                    //(CameraChangeMatrix4.M33) * (CameraChangeMatrix4.M33)
                    //)
                    //));

                    //sensor.ReconSensorControl.AngleZ = 180 / Math.PI * Math.Atan2(sensor.ReconCamera.WorldToCameraMatrix4.M32, CameraChangeMatrix4.M33);
                }

                tracked = false;
            }
            else
            {
                Matrix4 newView = new Matrix4();
                sensor.ReconCamera.RetreiveFrustumTransformMatrix4(newView);
                this.integrationWeight = this.integrationWeight--;
                if (this.integrationWeight < 1)
                {
                    this.integrationWeight = 1;
                }
                sensor.ReconCamera.WorldToCameraMatrix4 = this.volume.GetCurrentWorldToCameraTransform();
                sensor.ReconSensorControl.CameraTransformMatrix = this.volume.GetCurrentWorldToCameraTransform();
            }
        }

        private bool first = true;
        private void ReconstructDepthData(ReconstructionSensor sensor)
        {
            try
            {
                if (null != this.volume && !this.savingMesh)
                {
                    this.Dispatcher.BeginInvoke(
                        (Action)(() =>
                            {
                                this.statusBarText.Text = Properties.Resources.ReconstructFrame
                                                          + this.processedFrameCount.ToString(CultureInfo.CurrentCulture);
                            }));

                    // Process and display depth data
                    this.PreProcessDepthData(sensor);

                    // We would do camera tracking here if required...
                   

                    // Lock the volume operations
                    lock (this.reconstructionLock)
                    {
                        // Integrate the frame to volume
                        if (!this.PauseIntegration)
                        {
                            // Map color to depth if we want to integrate color too
                            if (sensor.CaptureColor && null != sensor.MappedColorFrame)
                            {
                                // Pre-process color
                                sensor.MapColorToDepth();

                                Dispatcher.BeginInvoke(
                                    (Action)
                                    (() =>
                                     this.volume.IntegrateFrame(
                                         sensor.DepthFloatFrame,
                                         sensor.MappedColorFrame,
                                         this.integrationWeight,
                                         FusionDepthProcessor.DefaultColorIntegrationOfAllAngles,
                                         sensor.ReconCamera.WorldToCameraMatrix4)));

                                // Flag that we have captured color
                                this.colorCaptured = true;

                            }
                            else if (this.tracking)
                            {
                                if (first)
                                {
                                    float centreDepth = this.depthFloatFrameDepthPixels[this.depthHeight * this.depthWidth / 2];
                                    if (centreDepth > 0.0)
                                    {
                                    //    Dispatcher.BeginInvoke(
                                    //(Action)
                                    //(() =>
                                    //    sensor.ReconSensorControl.AxisDistance = centreDepth
                                    //    ));
                                    }
                                    first = false;
                                }
                                Dispatcher.BeginInvoke(
                                    (Action)
                                    (() =>
                                        doDepthStuff(sensor)
                                ));
                            }
                            else
                            {
                                first = true;
                                // Just integrate depth
                                Dispatcher.BeginInvoke(
                                    (Action)
                                    (() =>
                                        this.volume.IntegrateFrame(
                                            sensor.DepthFloatFrame, this.integrationWeight, sensor.ReconCamera.WorldToCameraMatrix4)));

                            }
                        }
                    }
                    if (this.processedFrameCount % 2 == 0)
                    {
                        Dispatcher.BeginInvoke((Action)(() => this.RenderReconstruction(this.useCameraViewInReconstruction ? sensor : null)));
                    }
                }
            }
            catch (InvalidOperationException ex)
            {
                this.ShowStatusMessage(ex.Message);
            }
        }

        /// <summary>
        /// Render the reconstruction, optionally from the virtualCamera viewpoint
        /// </summary>
        /// <param name="sensor">Optionally, the the sensor to use for reconstruction rendering viewpoint, or null to use the graphics camera pose.</param>
        private void RenderReconstruction(ReconstructionSensor sensor)
        {
            try
            {
                Matrix4 cameraView = (sensor == null) ? this.virtualCamera.WorldToCameraMatrix4 : sensor.ReconCamera.WorldToCameraMatrix4;

                if (null != this.volume && !this.savingMesh && null != this.PointCloudFrame && null != this.ShadedSurfaceFrame
                    && null != this.ShadedSurfaceNormalsFrame)
                {
                    // Lock the volume operations
                    lock (this.reconstructionLock)
                    {
                        bool colorInUse = false;

                        // Calculate the point cloud of integration and optionally return the integrated color
                        foreach (ReconstructionSensor individualSensors in this.sensors)
                        {
                            // Take first sensor which is actually in use
                            if (individualSensors.UseSensor && individualSensors.CaptureColor)
                            {
                                colorInUse = true;
                                break;
                            }
                        }

                        if (this.colorCaptured && colorInUse)
                        {
                            this.volume.CalculatePointCloud(this.PointCloudFrame, this.ShadedSurfaceFrame, cameraView);
                        }
                        else
                        {
                            this.volume.CalculatePointCloud(this.PointCloudFrame, cameraView);

                            // Shade point cloud frame for rendering
                            FusionDepthProcessor.ShadePointCloud(
                                this.PointCloudFrame, cameraView, this.ShadedSurfaceFrame, this.ShadedSurfaceNormalsFrame);
                        }
                    }

                    // Run the UI update
                    Dispatcher.BeginInvoke((Action)(() => this.ReconstructFrameComplete(sensor)));
                }
            }
            catch (InvalidOperationException ex)
            {
                this.ShowStatusMessage(ex.Message);
            }
        }

        /// <summary>
        /// Called when a depth frame is available for display in the UI 
        /// </summary>
        /// <param name="sensor">The the sensor in use</param>
        private void DepthFrameComplete(ReconstructionSensor sensor)
        {
            if (this.firstFrame)
            {
                this.firstFrame = false;

                // Render shaded surface frame or shaded surface normals frame - blank at this point
                RenderColorImage(
                    this.sensors[0].ShadedSurfaceFrame,
                    ref this.shadedSurfaceFramePixelsArgb,
                    ref this.shadedSurfaceFrameBitmap,
                    this.shadedSurfaceImage);
            }

            // Render depth float frame
            this.RenderDepthFloatImage(sensor.DepthFloatFrame, ref this.depthFloatFrameBitmap, this.depthFloatImage);
        }

        /// <summary>
        /// Called when a ray-casted view of the reconstruction is available for display in the UI 
        /// </summary>
        /// <param name="sensor">The the sensor in use</param>
        private void ReconstructFrameComplete(ReconstructionSensor sensor)
        {
            // Render shaded surface frame or shaded surface normals frame
            if (sensor == null)
            {
                // Use Graphics camera
                RenderColorImage(
                    this.displayNormals ? this.ShadedSurfaceNormalsFrame : this.ShadedSurfaceFrame,
                    ref this.shadedSurfaceFramePixelsArgb,
                    ref this.shadedSurfaceFrameBitmap,
                    this.shadedSurfaceImage);
            }
            else
            {
                RenderColorImage(
                    this.displayNormals ? sensor.ShadedSurfaceNormalsFrame : sensor.ShadedSurfaceFrame,
                    ref this.shadedSurfaceFramePixelsArgb,
                    ref this.shadedSurfaceFrameBitmap,
                    this.shadedSurfaceImage);
            }
        }

        /// <summary>
        /// Create a WorldToVolume transform which sets the world origin at the center of the volume
        /// </summary>
        /// <returns>A Matrix4 containing the world-to-volume transform.</returns>
        private Matrix4 CreateWorldToVolumeTransform()
        {
            Matrix4 worldToVolume = Matrix4.Identity;

            worldToVolume.M11 = this.voxelsPerMeter;
            worldToVolume.M22 = this.voxelsPerMeter;
            worldToVolume.M33 = this.voxelsPerMeter;

            worldToVolume.M41 = this.voxelsX / 2;
            worldToVolume.M42 = this.voxelsY / 2;
            worldToVolume.M43 = this.voxelsZ / 2;

            return worldToVolume;
        }

        /// <summary>
        /// Reset reconstruction object to initial state
        /// </summary>
        /// <param name="sender">Event generator</param>
        /// <param name="e">Event parameter</param>
        private void ResetReconstruction(object sender, EventArgs e)
        {
            if (null == this.sensors)
            {
                return;
            }

            // Reset volume
            if (null != this.volume)
            {
                try
                {
                    this.worldToVolumeTransform = this.CreateWorldToVolumeTransform();

                    this.volume.ResetReconstruction(Matrix4.Identity, this.worldToVolumeTransform);

                    if (this.PauseIntegration)
                    {
                        this.PauseIntegration = false;
                    }
                }
                catch (InvalidOperationException)
                {
                    this.ShowStatusMessage(Properties.Resources.ResetFailed);
                }

                this.DoneReconstructing();
            }

            this.firstFrame = true;
            this.colorCaptured = false;

            // Reset fps counter
            this.ResetFps();
        }

        /// <summary>
        /// Re-create the reconstruction object
        /// </summary>
        /// <returns>Indicate success or failure</returns>
        private bool RecreateReconstruction()
        {
            // Check if sensors has been initialized
            if (null == this.sensors)
            {
                return false;
            }

            if (null != this.volume)
            {
                lock (this.reconstructionLock)
                {
                    this.volume.Dispose();
                }
            }

            try
            {
                // The zero-based GPU index to choose for reconstruction processing if the 
                // ReconstructionProcessor AMP options are selected.
                // Here we automatically choose a device to use for processing by passing -1, 
                int deviceIndex = -1;

                ReconstructionParameters volParam = new ReconstructionParameters(
                    this.voxelsPerMeter, this.voxelsX, this.voxelsY, this.voxelsZ);

                // Here we set internal camera pose to identity, as we mange each separately in the ReconstructionSensor class
                this.volume = ColorReconstruction.FusionCreateReconstruction(volParam, ProcessorType, deviceIndex, Matrix4.Identity);

                // We need to call reset here to set the correct world-to-volume transform
                this.ResetReconstruction(this, null);

                // Reset "Pause Integration"
                if (this.PauseIntegration)
                {
                    this.PauseIntegration = false;
                }

                this.firstFrame = true;
                this.DoneReconstructing();

                // Create volume cube 3D graphics in WPF3D. The front top left corner is the actual origin of the volume
                // voxel coordinate system, and shown with an overlaid coordinate cross.
                // Red is the +X axis, Green is the +Y axis, Blue is the +Z axis in the voxel coordinate system
                this.DisposeVolumeCube3DGraphics(); // Auto-removes from the visual tree
                this.CreateCube3DGraphics(volumeCubeLineColor, LineThickness, new Vector3D(0, 0, this.worldToVolumeTransform.M43 / this.voxelsPerMeter)); // Auto-adds to the visual tree
                this.AddVolumeCube3DGraphics();

                return true;
            }
            catch (ArgumentException)
            {
                this.volume = null;
                this.ShowStatusMessage(Properties.Resources.VolumeResolution);
            }
            catch (InvalidOperationException ex)
            {
                this.volume = null;
                this.ShowStatusMessage(ex.Message);
            }
            catch (DllNotFoundException)
            {
                this.volume = null;
                this.ShowStatusMessage(Properties.Resources.MissingPrerequisite);
            }
            catch (OutOfMemoryException)
            {
                this.volume = null;
                this.ShowStatusMessage(Properties.Resources.OutOfMemory);
            }

            return false;
        }

        /// <summary>
        /// Handler for click event from "Reconstruct Frames" button
        /// </summary>
        /// <param name="sender">Event sender</param>
        /// <param name="e">Event arguments</param>
        private void ReconstructFramesButtonClick(object sender, RoutedEventArgs e)
        {
            if (null == this.sensors)
            {
                return;
            }

            // Reconstruct frames
            bool[] successfulSaveSettings = this.Reconstruct();

            if (null == successfulSaveSettings)
            {
                this.lastSensorSettingStatus = string.Empty;
            }
            else
            {
                StringBuilder successfulSaveCameras = new StringBuilder("Camera ");
                StringBuilder failedSaveCameras = new StringBuilder("Camera ");
                bool hasSuccessfulSavedCamera = false;
                bool hasFailedSavedCamera = false;

                for (int i = 0; i < successfulSaveSettings.Length; i++)
                {
                    if (successfulSaveSettings[i])
                    {
                        successfulSaveCameras.AppendFormat("{0} ", i);

                        hasSuccessfulSavedCamera = true;
                    }
                    else
                    {
                        failedSaveCameras.AppendFormat("{0} ", i);

                        hasFailedSavedCamera = true;
                    }
                }

                StringBuilder cameraSavingStatus = new StringBuilder(string.Empty);
                if (hasSuccessfulSavedCamera)
                {
                    cameraSavingStatus.AppendFormat("{0}: {1}    ", successfulSaveCameras, Properties.Resources.SettingsSaved);
                }

                if (hasFailedSavedCamera)
                {
                    cameraSavingStatus.AppendFormat("{0}: {1}    ", failedSaveCameras, Properties.Resources.ErrorSaveSettings);
                }

                this.lastSensorSettingStatus = cameraSavingStatus.ToString();
            }

            // Update manual reset information to status bar
            this.ShowStatusMessage(Properties.Resources.Reconstructing);
        }

        /// <summary>
        /// Handler for click event from "Reset Reconstruction" button
        /// </summary>
        /// <param name="sender">Event sender</param>
        /// <param name="e">Event arguments</param>
        private void ResetReconstructionButtonClick(object sender, RoutedEventArgs e)
        {
            if (null == this.sensors)
            {
                return;
            }

            // Reset volume
            this.ResetReconstruction(this, null);

            // Update manual reset information to status bar
            this.ShowStatusMessage(Properties.Resources.ResetVolume);
        }

        /// <summary>
        /// Handler for click event from "Create Mesh" button
        /// </summary>
        /// <param name="sender">Event sender.</param>
        /// <param name="e">Event arguments.</param>
        private void CreateMeshButtonClick(object sender, RoutedEventArgs e)
        {
            if (null == this.sensors)
            {
                return;
            }

            if (tracking)
            {

                this.tracking = false;
                bool[] successfulSave = null;
                if (this.loadSavePerKinectSettings && this.sensors.Count > 0)
                {
                    successfulSave = new bool[this.sensors.Count];

                    for (int i = 0; i < this.sensors.Count; i++)
                    {
                        successfulSave[i] = this.sensors[i].SaveSettings();
                    }
                }
            }
            else
            {
                tracking = true;
                // Reconstruct frames
                bool[] successfulSaveSettings = this.Reconstruct();

                if (null == successfulSaveSettings)
                {
                    this.lastSensorSettingStatus = string.Empty;
                }
                else
                {
                    StringBuilder successfulSaveCameras = new StringBuilder("Camera ");
                    StringBuilder failedSaveCameras = new StringBuilder("Camera ");
                    bool hasSuccessfulSavedCamera = false;
                    bool hasFailedSavedCamera = false;

                    for (int i = 0; i < successfulSaveSettings.Length; i++)
                    {
                        if (successfulSaveSettings[i])
                        {
                            successfulSaveCameras.AppendFormat("{0} ", i);

                            hasSuccessfulSavedCamera = true;
                        }
                        else
                        {
                            failedSaveCameras.AppendFormat("{0} ", i);

                            hasFailedSavedCamera = true;
                        }
                    }

                    StringBuilder cameraSavingStatus = new StringBuilder(string.Empty);
                    if (hasSuccessfulSavedCamera)
                    {
                        cameraSavingStatus.AppendFormat("{0}: {1}    ", successfulSaveCameras, Properties.Resources.SettingsSaved);
                    }

                    if (hasFailedSavedCamera)
                    {
                        cameraSavingStatus.AppendFormat("{0}: {1}    ", failedSaveCameras, Properties.Resources.ErrorSaveSettings);
                    }

                    this.lastSensorSettingStatus = cameraSavingStatus.ToString();
                }

                // Update manual reset information to status bar
                this.ShowStatusMessage(Properties.Resources.Reconstructing);
            }
        }

        //private void CreateMeshButtonClick(object sender, RoutedEventArgs e)
        //{
        //    if (null == this.volume)
        //    {
        //        this.ShowStatusMessage(Properties.Resources.MeshNullVolume);
        //        return;
        //    }

        //    this.savingMesh = true;

        //    try
        //    {
        //        this.ShowStatusMessage(Properties.Resources.SavingMesh);

        //        ColorMesh mesh = this.volume.CalculateMesh(1);

        //        Win32.SaveFileDialog dialog = new Win32.SaveFileDialog();

        //        if (true == this.stlFormat.IsChecked)
        //        {
        //            dialog.FileName = "MeshedReconstruction.stl";
        //            dialog.Filter = "STL Mesh Files|*.stl|All Files|*.*";
        //        }
        //        else if (true == this.objFormat.IsChecked)
        //        {
        //            dialog.FileName = "MeshedReconstruction.obj";
        //            dialog.Filter = "OBJ Mesh Files|*.obj|All Files|*.*";
        //        }
        //        else
        //        {
        //            dialog.FileName = "MeshedReconstruction.ply";
        //            dialog.Filter = "PLY Mesh Files|*.ply|All Files|*.*";
        //        }

        //        if (true == dialog.ShowDialog())
        //        {
        //            if (true == this.stlFormat.IsChecked)
        //            {
        //                using (BinaryWriter writer = new BinaryWriter(dialog.OpenFile()))
        //                {
        //                    // Default to flip Y,Z coordinates on save
        //                    Helper.SaveBinaryStlMesh(mesh, writer, true);
        //                }
        //            }
        //            else if (true == this.objFormat.IsChecked)
        //            {
        //                using (StreamWriter writer = new StreamWriter(dialog.FileName))
        //                {
        //                    // Default to flip Y,Z coordinates on save
        //                    Helper.SaveAsciiObjMesh(mesh, writer, true);
        //                }
        //            }
        //            else
        //            {
        //                using (StreamWriter writer = new StreamWriter(dialog.FileName))
        //                {
        //                    // Default to flip Y,Z coordinates on save
        //                    Helper.SaveAsciiPlyMesh(mesh, writer, true, this.colorCaptured);
        //                }
        //            }

        //            this.ShowStatusMessage(Properties.Resources.MeshSaved);
        //        }
        //        else
        //        {
        //            this.ShowStatusMessage(Properties.Resources.MeshSaveCanceled);
        //        }
        //    }
        //    catch (ArgumentException)
        //    {
        //        this.ShowStatusMessage(Properties.Resources.ErrorSaveMesh);
        //    }
        //    catch (InvalidOperationException)
        //    {
        //        this.ShowStatusMessage(Properties.Resources.ErrorSaveMesh);
        //    }
        //    catch (IOException)
        //    {
        //        this.ShowStatusMessage(Properties.Resources.ErrorSaveMesh);
        //    }

        //    this.savingMesh = false;
        //}

        /// <summary>
        /// Handler for volume setting changing event
        /// </summary>
        /// <param name="sender">Event sender</param>
        /// <param name="e">Event argument</param>
        private void VolumeSettingsChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (null != this.sensors && 0 < this.sensors.Count)
            {
                this.RecreateReconstruction();
            }
        }

        /// <summary>
        /// Turn sensor Laser emitter on or off, silently ignoring any errors.
        /// Note that errors will occur if a Kinect for Xbox sensor is attached as this
        /// does not support controlling the Laser emitter.
        /// </summary>
        /// <param name="sensor">The reconstruction sensor.</param>
        /// <param name="forceOff">Set true to turn off the sensor, false to turn sensor on.</param>
        private void ChangeSensorEmitterState(ReconstructionSensor sensor, bool forceOff)
        {
            if (null != sensor && null != sensor.Sensor)
            {
                try
                {
 //                   sensor.Sensor.ForceInfraredEmitterOff = forceOff;
                }
                catch (InvalidOperationException)
                {
                    // Fail silently
                }
            }
        }

        /// <summary>
        /// Show exception info on status bar
        /// </summary>
        /// <param name="message">Message to show on status bar</param>
        private void ShowStatusMessage(string message)
        {
            this.Dispatcher.BeginInvoke(
                (Action)(() =>
                    {
                        this.ResetFps();

                        if ((DateTime.Now - this.lastStatusTimestamp).Seconds >= StatusBarInterval)
                        {
                            this.statusBarText.Text = message;
                        }
                        else
                        {
                            this.statusMessageQueue.Enqueue(message);
                        }

                        this.lastStatusTimestamp = DateTime.Now;
                    }));
        }

        /// <summary>
        /// Render Fusion depth float frame to UI
        /// </summary>
        /// <param name="depthFloatFrame">Fusion depth float frame</param>
        /// <param name="bitmap">Bitmap contains depth float frame data for rendering</param>
        /// <param name="image">UI image component to render depth float frame to</param>
        private void RenderDepthFloatImage(
            FusionFloatImageFrame depthFloatFrame, ref WriteableBitmap bitmap, System.Windows.Controls.Image image)
        {
            if (null == depthFloatFrame)
            {
                return;
            }

            // PixelDataLength is the number of pixels, not bytes
            if (null == this.depthFloatFramePixelsArgb || depthFloatFrame.PixelDataLength != this.depthFloatFramePixelsArgb.Length)
            {
                // Create colored pixel array of correct format
                this.depthFloatFramePixelsArgb = new int[depthFloatFrame.PixelDataLength];
            }

            if (null == this.depthFloatFrameDepthPixels || depthFloatFrame.PixelDataLength != this.depthFloatFrameDepthPixels.Length)
            {
                // Create colored pixel array of correct format
                this.depthFloatFrameDepthPixels = new float[depthFloatFrame.PixelDataLength];
            }

            if (null == bitmap || depthFloatFrame.Width != bitmap.Width || depthFloatFrame.Height != bitmap.Height)
            {
                // Create bitmap of correct format
                bitmap = new WriteableBitmap(depthFloatFrame.Width, depthFloatFrame.Height, 96.0, 96.0, PixelFormats.Bgr32, null);

                // Set bitmap as source to UI image object
                image.Source = bitmap;
            }

            depthFloatFrame.CopyPixelDataTo(this.depthFloatFrameDepthPixels);

            // Calculate color of pixels based on depth of each pixel
            float range = 4.0f;
            float oneOverRange = 1.0f / range;
            float minRange = 0.0f;

            Parallel.For(
                0,
                depthFloatFrame.Height,
                y =>
                    {
                        int index = y * depthFloatFrame.Width;
                        for (int x = 0; x < depthFloatFrame.Width; ++x, ++index)
                        {
                            float depth = this.depthFloatFrameDepthPixels[index];
                            int intensity = (depth >= minRange) ? ((int)(((depth - minRange) * oneOverRange) * 256.0f) % 256) : 0;

                            this.depthFloatFramePixelsArgb[index] = (intensity << 16) | (intensity << 8) | intensity; // argb
                        }
                    });

            // Copy colored pixels to bitmap
            bitmap.WritePixels(
                new Int32Rect(0, 0, depthFloatFrame.Width, depthFloatFrame.Height),
                this.depthFloatFramePixelsArgb,
                bitmap.PixelWidth * sizeof(int),
                0);
        }

        /// <summary>
        /// Set the camera selected variable based on tab index
        /// </summary>
        /// <param name="sender">Event sender.</param>
        /// <param name="e">Event arguments.</param>
        private void OnSelectedTab(object sender, RoutedEventArgs e)
        {
            TabControl t = sender as TabControl;
            if (t != null)
            {
                this.cameraTabSelected = t.SelectedIndex;
            }
        }

        /// <summary>
        /// Reset the camera to the first camera pose
        /// </summary>
        /// <param name="sender">Event sender.</param>
        /// <param name="e">Event arguments.</param>
        private void ResetCameraButtonClick(object sender, RoutedEventArgs e)
        {
            if (null == this.virtualCamera)
            {
                return;
            }

            if (this.resetToFirstValidCameraView)
            {
                if (this.sensors.Count > 0)
                {
                    foreach (ReconstructionSensor sensor in this.sensors)
                    {
                        // Take first sensor which is actually in use
                        if (sensor.UseSensor)
                        {
                            this.virtualCamera.Reset();
                            this.virtualCamera.WorldToCameraMatrix3D = sensor.ReconCamera.WorldToCameraMatrix3D;

                            this.viewChanged = true;
                            break;
                        }
                    }
                }
            }
            else
            {
                this.virtualCamera.Reset();
                this.viewChanged = true;
            }
        }

        /// <summary>
        /// Create an axis-aligned coordinate cross for rendering in the WPF3D coordinate system. 
        /// Red is the +X axis, Green is the +Y axis, Blue is the +Z axis
        /// </summary>
        /// <param name="crossOrigin">The origin of the coordinate cross in world space.</param>
        /// <param name="axisSize">The size of the axis in m.</param>
        /// <param name="thickness">The thickness of the lines in screen pixels.</param>
        private void CreateAxisAlignedCoordinateCross3DGraphics(Point3D crossOrigin, float axisSize, int thickness)
        {
            this.axisX = new ScreenSpaceLines3D();

            this.axisX.Points = new Point3DCollection();
            this.axisX.Points.Add(crossOrigin);
            this.axisX.Points.Add(new Point3D(crossOrigin.X + axisSize, crossOrigin.Y, crossOrigin.Z));

            this.axisX.Thickness = 2;
            this.axisX.Color = System.Windows.Media.Color.FromArgb(200, 255, 0, 0); // Red (X)

            this.axisY = new ScreenSpaceLines3D();

            this.axisY.Points = new Point3DCollection();
            this.axisY.Points.Add(crossOrigin);
            this.axisY.Points.Add(new Point3D(crossOrigin.X, crossOrigin.Y + axisSize, crossOrigin.Z));

            this.axisY.Thickness = 2;
            this.axisY.Color = System.Windows.Media.Color.FromArgb(200, 0, 255, 0); // Green (Y)

            this.axisZ = new ScreenSpaceLines3D();

            this.axisZ.Points = new Point3DCollection();
            this.axisZ.Points.Add(crossOrigin);
            this.axisZ.Points.Add(new Point3D(crossOrigin.X, crossOrigin.Y, crossOrigin.Z + axisSize));

            this.axisZ.Thickness = thickness;
            this.axisZ.Color = System.Windows.Media.Color.FromArgb(200, 0, 0, 255); // Blue (Z)
        }

        /// <summary>
        /// Add the coordinate cross axes to the visual tree
        /// </summary>
        private void AddAxisAlignedCoordinateCross3DGraphics()
        {
            if (this.haveAddedCoordinateCross)
            {
                return;
            }

            if (null != this.axisX)
            {
                this.GraphicsViewport.Children.Add(this.axisX);

                this.haveAddedCoordinateCross = true;
            }

            if (null != this.axisY)
            {
                this.GraphicsViewport.Children.Add(this.axisY);
            }

            if (null != this.axisZ)
            {
                this.GraphicsViewport.Children.Add(this.axisZ);
            }
        }

        /// <summary>
        /// Remove the coordinate cross axes from the visual tree
        /// </summary>
        private void RemoveAxisAlignedCoordinateCross3DGraphics()
        {
            if (null != this.axisX)
            {
                this.GraphicsViewport.Children.Remove(this.axisX);
            }

            if (null != this.axisY)
            {
                this.GraphicsViewport.Children.Remove(this.axisY);
            }

            if (null != this.axisZ)
            {
                this.GraphicsViewport.Children.Remove(this.axisZ);
            }

            this.haveAddedCoordinateCross = false;
        }

        /// <summary>
        /// Dispose the coordinate cross axes from the visual tree
        /// </summary>
        private void DisposeAxisAlignedCoordinateCross3DGraphics()
        {
            if (this.haveAddedCoordinateCross)
            {
                this.RemoveAxisAlignedCoordinateCross3DGraphics();
            }

            if (null != this.axisX)
            {
                this.axisX.Dispose();
                this.axisX = null;
            }

            if (null != this.axisY)
            {
                this.axisY.Dispose();
                this.axisY = null;
            }

            if (null != this.axisZ)
            {
                this.axisZ.Dispose();
                this.axisZ = null;
            }
        }

        /// <summary>
        /// Create an axis-aligned volume cube for rendering.
        /// </summary>
        /// <param name="color">The color of the volume cube.</param>
        /// <param name="thickness">The thickness of the lines in screen pixels.</param>
        /// <param name="translation">World to volume translation vector.</param>
        private void CreateCube3DGraphics(System.Windows.Media.Color color, int thickness, Vector3D translation)
        {
            // Scaler for cube size
            float cubeSizeScaler = 1.0f;

            // Before we created a volume which contains the head
            // Here we create a graphical representation of this volume cube
            float oneOverVpm = 1.0f / this.voxelsPerMeter;

            // This cube is world axis aligned
            float cubeSideX = this.voxelsX * oneOverVpm * cubeSizeScaler;
            float halfSideX = cubeSideX * 0.5f;

            float cubeSideY = this.voxelsY * oneOverVpm * cubeSizeScaler;
            float halfSideY = cubeSideY * 0.5f;

            float cubeSideZ = this.voxelsZ * oneOverVpm * cubeSizeScaler;
            float halfSideZ = cubeSideZ * 0.5f;

            // The translation vector is from the origin to the volume front face
            // And here we describe the translation Z as from the origin to the cube center
            // So we continue to translate half volume size align Z
            translation.Z -= halfSideZ / cubeSizeScaler;

            this.volumeCube = new ScreenSpaceLines3D();
            this.volumeCube.Points = new Point3DCollection();

            // Front face
            // TL front - TR front
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, -halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, halfSideY + translation.Y, -halfSideZ + translation.Z));

            // TR front - BR front
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, halfSideY + translation.Y, -halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, -halfSideY + translation.Y, -halfSideZ + translation.Z));

            // BR front - BL front
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, -halfSideY + translation.Y, -halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, -halfSideY + translation.Y, -halfSideZ + translation.Z));

            // BL front - TL front
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, -halfSideY + translation.Y, -halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, -halfSideZ + translation.Z));

            // Rear face
            // TL rear - TR rear
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));

            // TR rear - BR rear
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, -halfSideY + translation.Y, halfSideZ + translation.Z));

            // BR rear - BL rear
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, -halfSideY + translation.Y, halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, -halfSideY + translation.Y, halfSideZ + translation.Z));

            // BL rear - TL rear
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, -halfSideY + translation.Y, halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));

            // Connecting lines
            // TL front - TL rear
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, -halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));

            // TR front - TR rear
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, halfSideY + translation.Y, -halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));

            // BR front - BR rear
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, -halfSideY + translation.Y, -halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(halfSideX + translation.X, -halfSideY + translation.Y, halfSideZ + translation.Z));

            // BL front - BL rear
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, -halfSideY + translation.Y, -halfSideZ + translation.Z));
            this.volumeCube.Points.Add(new Point3D(-halfSideX + translation.X, -halfSideY + translation.Y, halfSideZ + translation.Z));

            this.volumeCube.Thickness = thickness;
            this.volumeCube.Color = color;

            this.volumeCubeAxisX = new ScreenSpaceLines3D();

            this.volumeCubeAxisX.Points = new Point3DCollection();
            this.volumeCubeAxisX.Points.Add(new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));
            this.volumeCubeAxisX.Points.Add(
                new Point3D(-halfSideX + 0.1f + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));

            this.volumeCubeAxisX.Thickness = thickness + 2;
            this.volumeCubeAxisX.Color = System.Windows.Media.Color.FromArgb(200, 255, 0, 0); // Red (X)

            this.volumeCubeAxisY = new ScreenSpaceLines3D();

            this.volumeCubeAxisY.Points = new Point3DCollection();
            this.volumeCubeAxisY.Points.Add(new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));
            this.volumeCubeAxisY.Points.Add(
                new Point3D(-halfSideX + translation.X, halfSideY - 0.1f + translation.Y, halfSideZ + translation.Z));

            this.volumeCubeAxisY.Thickness = thickness + 2;
            this.volumeCubeAxisY.Color = System.Windows.Media.Color.FromArgb(200, 0, 255, 0); // Green (Y)

            this.volumeCubeAxisZ = new ScreenSpaceLines3D();

            this.volumeCubeAxisZ.Points = new Point3DCollection();
            this.volumeCubeAxisZ.Points.Add(new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, halfSideZ + translation.Z));
            this.volumeCubeAxisZ.Points.Add(
                new Point3D(-halfSideX + translation.X, halfSideY + translation.Y, halfSideZ - 0.1f + translation.Z));

            this.volumeCubeAxisZ.Thickness = thickness + 2;
            this.volumeCubeAxisZ.Color = System.Windows.Media.Color.FromArgb(200, 0, 0, 255); // Blue (Z)
        }

        /// <summary>
        /// Add the volume cube and axes to the visual tree
        /// </summary>
        private void AddVolumeCube3DGraphics()
        {
            if (this.haveAddedVolumeCube)
            {
                return;
            }

            if (null != this.volumeCube)
            {
                this.GraphicsViewport.Children.Add(this.volumeCube);

                this.haveAddedVolumeCube = true;
            }

            if (null != this.volumeCubeAxisX)
            {
                this.GraphicsViewport.Children.Add(this.volumeCubeAxisX);
            }

            if (null != this.volumeCubeAxisY)
            {
                this.GraphicsViewport.Children.Add(this.volumeCubeAxisY);
            }

            if (null != this.volumeCubeAxisZ)
            {
                this.GraphicsViewport.Children.Add(this.volumeCubeAxisZ);
            }
        }

        /// <summary>
        /// Remove the volume cube and axes from the visual tree
        /// </summary>
        private void RemoveVolumeCube3DGraphics()
        {
            if (null != this.volumeCube)
            {
                this.GraphicsViewport.Children.Remove(this.volumeCube);
            }

            if (null != this.volumeCubeAxisX)
            {
                this.GraphicsViewport.Children.Remove(this.volumeCubeAxisX);
            }

            if (null != this.volumeCubeAxisY)
            {
                this.GraphicsViewport.Children.Remove(this.volumeCubeAxisY);
            }

            if (null != this.volumeCubeAxisZ)
            {
                this.GraphicsViewport.Children.Remove(this.volumeCubeAxisZ);
            }

            this.haveAddedVolumeCube = false;
        }

        /// <summary>
        /// Dispose the volume cube and axes
        /// </summary>
        private void DisposeVolumeCube3DGraphics()
        {
            if (this.haveAddedVolumeCube)
            {
                this.RemoveVolumeCube3DGraphics();
            }

            if (null != this.volumeCube)
            {
                this.volumeCube.Dispose();
                this.volumeCube = null;
            }

            if (null != this.volumeCubeAxisX)
            {
                this.volumeCubeAxisX.Dispose();
                this.volumeCubeAxisX = null;
            }

            if (null != this.volumeCubeAxisY)
            {
                this.volumeCubeAxisY.Dispose();
                this.volumeCubeAxisY = null;
            }

            if (null != this.volumeCubeAxisZ)
            {
                this.volumeCubeAxisZ.Dispose();
                this.volumeCubeAxisZ = null;
            }
        }
    }

    /// <summary>
    /// Convert depth to UI text for meters distance
    /// </summary>
    public class DepthToTextConverter : IValueConverter
    {
        /// <summary>
        /// Length of converter parameter that as an array
        /// </summary>
        private int converterArrayLength = 2;

        /// <summary>
        /// Convert float depth value to text
        /// </summary>
        /// <param name="value">The object to convert.</param>
        /// <param name="targetType">The target type.</param>
        /// <param name="parameter">The parameter object.</param>
        /// <param name="culture">The conversion globalization information.</param>
        /// <returns>Returns the value converted to text.</returns>
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return ((double)value).ToString("0.00", CultureInfo.CurrentCulture);
        }

        /// <summary>
        /// Convert text to float depth value, clamping to a proper range
        /// </summary>
        /// <param name="value">The object to convert.</param>
        /// <param name="targetType">The target type.</param>
        /// <param name="parameter">The parameter object.</param>
        /// <param name="culture">The conversion globalization information.</param>
        /// <returns>Returns the text converted to a value.</returns>
        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            float rangeStart = FusionDepthProcessor.DefaultMinimumDepth;
            float rangeEnd = FusionDepthProcessor.DefaultMaximumDepth;
            double[] depthRange = parameter as double[];
            if (converterArrayLength == depthRange.Length)
            {
                rangeStart = (float)depthRange[0];
                rangeEnd = (float)depthRange[1];
            }

            float val = 0;
            float.TryParse(value as string, NumberStyles.AllowDecimalPoint | NumberStyles.AllowLeadingSign | NumberStyles.AllowLeadingWhite | NumberStyles.AllowTrailingWhite, CultureInfo.CurrentCulture, out val);

            val = val < rangeStart ? rangeStart : val;
            val = val > rangeEnd ? rangeEnd : val;

            return val;
        }
    }

    /// <summary>
    /// Convert integer to UI text for reconstruction frame count
    /// </summary>
    public class IntToTextConverter : IValueConverter
    {
        /// <summary>
        /// Convert int value to text
        /// </summary>
        /// <param name="value">The object to convert.</param>
        /// <param name="targetType">The target type.</param>
        /// <param name="parameter">The parameter object.</param>
        /// <param name="culture">The conversion globalization information.</param>
        /// <returns>Returns the value converted to text.</returns>
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return ((double)value).ToString("0", CultureInfo.CurrentCulture);
        }

        /// <summary>
        /// Convert text to int value, clamping to a proper range
        /// </summary>
        /// <param name="value">The object to convert.</param>
        /// <param name="targetType">The target type.</param>
        /// <param name="parameter">The parameter object.</param>
        /// <param name="culture">The conversion globalization information.</param>
        /// <returns>Returns the text converted to a value.</returns>
        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            int val = 0;
            int.TryParse(value as string, NumberStyles.Integer, CultureInfo.CurrentCulture, out val);

            val = val < MainWindow.MinReconstructionFrameInCount ? MainWindow.MinReconstructionFrameInCount : val;
            val = val > MainWindow.MaxReconstructionFrameInCount ? MainWindow.MaxReconstructionFrameInCount : val;

            return val;
        }
    }
}
