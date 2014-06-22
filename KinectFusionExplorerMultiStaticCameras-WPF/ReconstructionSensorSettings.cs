// -----------------------------------------------------------------------
// <copyright file="ReconstructionSensorSettings.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// -----------------------------------------------------------------------

namespace Microsoft.Samples.Kinect.KinectFusionExplorer
{
    using System;
    using Microsoft.Kinect;
    using Microsoft.Kinect.Toolkit;
    using Microsoft.Kinect.Toolkit.Fusion;

    /// <summary>
    /// Reconstruction Sensor Settings class
    /// </summary>
    [Serializable]
    public class ReconstructionSensorSettings
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ReconstructionSensorSettings"/> class.
        /// </summary>
        public ReconstructionSensorSettings()
        {
            this.UseSensor = true;
            this.NearMode = true;
            this.MirrorDepth = false;
            this.CaptureColor = false;
            this.MinDepthClip = FusionDepthProcessor.DefaultMinimumDepth;
            this.MaxDepthClip = FusionDepthProcessor.DefaultMaximumDepth;
            this.AngleX = 0;
            this.AngleY = 0;
            this.AngleZ = 0;
            this.AxisDistance = 0.50f;
            this.CameraTransformMatrix = new Matrix4();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ReconstructionSensorSettings"/> class.
        /// </summary>
        /// <param name="useSensor">True if the sensor is used for reconstruction.</param>
        /// <param name="nearMode">True if the sensor is using near mode.</param>
        /// <param name="mirrorDepth">True if the sensor is mirroring the depth image.</param>
        /// <param name="captureColor">True if the sensor is capturing color images.</param>
        /// <param name="minDepthClip">The near distance to clip the depth signal at. Closer distances will be set to 0.</param>
        /// <param name="maxDepthClip">The far distance to clip the depth signal at. Further distances will be set to 1000.</param>
        /// <param name="angleX">Rotation angle around X axis.</param>
        /// <param name="angleY">Rotation angle around Y axis.</param>
        /// <param name="angleZ">Rotation angle around Z axis.</param>
        /// <param name="axisDistance">Distance from world origin.</param>
        public ReconstructionSensorSettings(
            bool useSensor,
            bool nearMode,
            bool mirrorDepth,
            bool captureColor,
            float minDepthClip,
            float maxDepthClip,
            float angleX,
            float angleY,
            float angleZ,
            float axisDistance)
        {
            this.UseSensor = useSensor;
            this.NearMode = nearMode;
            this.MirrorDepth = mirrorDepth;
            this.CaptureColor = captureColor;
            this.MinDepthClip = minDepthClip;
            this.MaxDepthClip = maxDepthClip;
            this.AngleX = angleX;
            this.AngleY = angleY;
            this.AngleZ = angleZ;
            this.AxisDistance = axisDistance;
            this.CameraTransformMatrix = new Matrix4();
        }

        #region Properties

        /// <summary>
        /// Gets or sets a value indicating whether to use the Kinect sensor in reconstruction
        /// </summary>
        public bool UseSensor { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether near mode is enabled for the Kinect sensor
        /// </summary>
        public bool NearMode { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the depth image is mirrored
        /// </summary>
        public bool MirrorDepth { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to capture color
        /// </summary>
        public bool CaptureColor { get; set; }

        /// <summary>
        /// Gets or sets the minimum depth distance threshold in meters. Depth pixels below this value will be
        /// returned as invalid (0). Min depth must be positive or 0.
        /// </summary>
        public float MinDepthClip { get; set; }

        /// <summary>
        /// Gets or sets the maximum depth distance threshold in meters. Depth pixels above this value will be
        /// returned as invalid (0). Max depth must be greater than 0.
        /// </summary>
        public float MaxDepthClip { get; set; }

        /// <summary>
        /// Gets or sets the sensor rotation angle around X relative to world axes
        /// </summary>
        public float AngleX { get; set; }

        /// <summary>
        /// Gets or sets the sensor rotation angle around Y relative to world axes
        /// </summary>
        public float AngleY { get; set; }

        /// <summary>
        /// Gets or sets the sensor rotation angle around Z relative to world axes
        /// </summary>
        public float AngleZ { get; set; }

        /// <summary>
        /// Gets or sets the Kinect sensor distance relative to world origin
        /// </summary>
        public float AxisDistance { get; set; }

        public Matrix4 CameraTransformMatrix { get; set; }

        #endregion
    }
}
